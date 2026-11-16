"""Complete tiny positive-penalty LASSO optimizer face over stored inputs."""
from fractions import Fraction as F
from itertools import combinations
from .core import number, validate
from .oracle import enumerate_optima


def _rectangular(columns, target):
    """Unique full-column-rank solution of columns*z=target, or None.

    Eliminate the original rectangular design, not its Gram matrix. All rows
    including residual zero rows are checked. Empty columns admit only zero.
    """
    n, m = len(target), len(columns)
    rows = [[columns[j][i] for j in range(m)] + [target[i]] for i in range(n)]
    pivots = []
    cursor = 0
    for j in range(m):
        pivot = next((i for i in range(cursor, n) if rows[i][j]), None)
        if pivot is None:
            return None
        rows[cursor], rows[pivot] = rows[pivot], rows[cursor]
        scale = rows[cursor][j]
        rows[cursor] = [v/scale for v in rows[cursor]]
        for i in range(n):
            if i != cursor:
                scale = rows[i][j]
                rows[i] = [a-scale*b for a,b in zip(rows[i],rows[cursor])]
        pivots.append(j)
        cursor += 1
    if any(not any(row[:-1]) and row[-1] for row in rows):
        return None
    result = [rows[i][-1] for i in range(m)]
    if any(sum((columns[j][i]*result[j] for j in range(m)),F(0)) != target[i] for i in range(n)):
        return None
    return result


def _rank(rows):
    rows = [list(row) for row in rows]
    if not rows:
        return 0
    cursor = 0
    for j in range(len(rows[0])):
        pivot = next((i for i in range(cursor,len(rows)) if rows[i][j]),None)
        if pivot is None:
            continue
        rows[cursor],rows[pivot] = rows[pivot],rows[cursor]
        scale = rows[cursor][j]
        rows[cursor] = [v/scale for v in rows[cursor]]
        for i in range(cursor+1,len(rows)):
            scale = rows[i][j]
            rows[i] = [a-scale*b for a,b in zip(rows[i],rows[cursor])]
        cursor += 1
        if cursor == len(rows):
            break
    return cursor


def optimal_face(x, y, penalty):
    x,y = validate(x,y)
    if len(x)>32 or len(x[0])>5:
        raise ValueError('optimal face limited to 32 rows and 5 columns')
    lam = F(number(penalty,'penalty',1e-8,10000))
    seed_report = enumerate_optima(x,y,penalty)
    seed = seed_report['witnesses'][0]
    x = [[F(v) for v in row]for row in x]
    y = [F(v) for v in y]
    n,p = len(x),len(x[0])
    prediction = list(map(F,seed['prediction']))
    residual = [a-b for a,b in zip(y,prediction)]
    corr = [sum((row[j]*r for row,r in zip(x,residual)),F(0))/n for j in range(p)]
    equicorrelation = [j for j,c in enumerate(corr) if abs(c)==lam]
    signs = [1 if corr[j]>0 else -1 for j in equicorrelation]
    columns = [[sign*row[j] for row in x]for j,sign in zip(equicorrelation,signs)]
    vertices = set()
    subsets = []
    for size in range(len(equicorrelation)+1):
        for subset in combinations(range(len(equicorrelation)),size):
            z = _rectangular([columns[i]for i in subset],prediction)
            status = 'rank_deficient_or_inconsistent' if z is None else 'negative_coordinate' if any(v<0 for v in z) else 'feasible_vertex'
            subsets.append({'equicorrelation_positions':list(subset),'status':status,
                            'solution':None if z is None else [str(v)for v in z]})
            if status != 'feasible_vertex':
                continue
            beta = [F(0)]*p
            for k,v in zip(subset,z):
                beta[equicorrelation[k]]=signs[k]*v
            vertices.add(tuple(beta))
    vertices = sorted(vertices)
    assert vertices
    dual_vector = [r/n for r in residual]
    dual = sum((a*b for a,b in zip(y,dual_vector)),F(0))-n*sum((v*v for v in dual_vector),F(0))/2
    l1 = sum(abs(v)for v in vertices[0])
    witnesses = []
    for beta in vertices:
        pred = [sum((a*b for a,b in zip(row,beta)),F(0))for row in x]
        norm = sum(abs(v)for v in beta)
        primal = sum((r*r for r in residual),F(0))/(2*n)+lam*norm
        assert pred==prediction and norm==l1 and primal==dual
        assert all((c==lam if b>0 else c==-lam) if b else abs(c)<=lam for b,c in zip(beta,corr))
        witnesses.append({'coefficients':[str(v)for v in beta], 'prediction':[str(v)for v in pred],
                          'l1_norm':str(norm),'primal':str(primal),'dual_gap':str(primal-dual)})
    ranges = []
    for j in range(p):
        low = min(beta[j]for beta in vertices)
        high = max(beta[j]for beta in vertices)
        ranges.append({'index':j,'lower':str(low),'upper':str(high),
                       'lower_vertex':next(k for k,beta in enumerate(vertices)if beta[j]==low),
                       'upper_vertex':next(k for k,beta in enumerate(vertices)if beta[j]==high),
                       'always_zero':low==high==0,'always_nonzero':low>0 or high<0})
    center = [sum((beta[j]for beta in vertices),F(0))/len(vertices)for j in range(p)]
    dimension = _rank([[a-b for a,b in zip(beta,vertices[0])]for beta in vertices[1:]])
    return {'schema_version':1,'arithmetic':'exact rational representation of validated binary64 inputs',
            'X':[[str(v)for v in row]for row in x],'y':[str(v)for v in y],'penalty':str(lam),
            'prediction':[str(v)for v in prediction],'residual':[str(v)for v in residual],
            'correlations':[str(v)for v in corr],'dual_vector':[str(v)for v in dual_vector],'dual':str(dual),
            'l1_norm':str(l1),'equicorrelation':equicorrelation,'equicorrelation_signs':signs,
            'vertices':witnesses,'vertex_count':len(vertices),'dimension':dimension,
            'coefficient_ranges':ranges,'vertex_average':[str(v)for v in center],
            'unique':len(vertices)==1,'always_zero':[v['index']for v in ranges if v['always_zero']],
            'always_nonzero':[v['index']for v in ranges if v['always_nonzero']],
            'subset_witnesses':subsets,'seed_coefficients':seed['coefficients'],
            'scope':'complete convex hull of optimizers of the positive-penalty stored-input LASSO; finite exact coefficient ambiguity, not causal feature importance, statistical confidence or a scalable solver'}
