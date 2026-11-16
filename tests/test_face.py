import copy
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from sparselab import enumerate_optima, fit, screen
from sparselab.face import optimal_face, _rectangular, _rank


class FaceTests(unittest.TestCase):
    def check(self,r):
        x=[[F(v)for v in row]for row in r['X']];y=list(map(F,r['y']));lam=F(r['penalty']);n=len(x);p=len(x[0]);prediction=list(map(F,r['prediction']));residual=[a-b for a,b in zip(y,prediction)]
        self.assertEqual(list(map(F,r['residual'])),residual)
        corr=[sum(row[j]*v for row,v in zip(x,residual))/n for j in range(p)]
        self.assertEqual(list(map(F,r['correlations'])),corr)
        self.assertEqual(r['equicorrelation'],[j for j,c in enumerate(corr)if abs(c)==lam])
        self.assertEqual(r['equicorrelation_signs'],[1 if corr[j]>0 else -1 for j in r['equicorrelation']])
        dual=sum(a*b for a,b in zip(y,map(F,r['dual_vector'])))-n*sum(v*v for v in map(F,r['dual_vector']))/2
        self.assertEqual(F(r['dual']),dual)
        vectors=[]
        for vertex in r['vertices']:
            beta=list(map(F,vertex['coefficients']));vectors.append(beta)
            self.assertEqual([sum(a*b for a,b in zip(row,beta))for row in x],prediction)
            primal=sum(v*v for v in residual)/(2*n)+lam*sum(abs(v)for v in beta)
            self.assertEqual(primal,dual);self.assertEqual(F(vertex['primal']),primal);self.assertEqual(vertex['dual_gap'],'0')
            self.assertEqual(vertex['prediction'],r['prediction']);self.assertEqual(F(vertex['l1_norm']),sum(abs(v)for v in beta))
            for j,b in enumerate(beta):self.assertEqual(corr[j],lam if b>0 else -lam)if b else self.assertLessEqual(abs(corr[j]),lam)
        self.assertEqual(r['vertex_count'],len(vectors));self.assertEqual(r['unique'],len(vectors)==1)
        self.assertEqual(r['vertex_average'],[str(sum(beta[j]for beta in vectors)/len(vectors))for j in range(p)])
        for entry in r['coefficient_ranges']:
            j=entry['index'];lo,hi=F(entry['lower']),F(entry['upper'])
            self.assertEqual(lo,min(beta[j]for beta in vectors));self.assertEqual(hi,max(beta[j]for beta in vectors))
            self.assertEqual(vectors[entry['lower_vertex']][j],lo);self.assertEqual(vectors[entry['upper_vertex']][j],hi)
            self.assertEqual(entry['always_zero'],lo==hi==0);self.assertEqual(entry['always_nonzero'],lo>0 or hi<0)
        self.assertEqual(r['always_zero'],[e['index']for e in r['coefficient_ranges']if e['always_zero']])
        self.assertEqual(r['always_nonzero'],[e['index']for e in r['coefficient_ranges']if e['always_nonzero']])

    def test_duplicate_continuum_and_attained_coordinate_ranges(self):
        x=[[1,1,1],[1,1,-1],[-1,-1,1],[-1,-1,-1]];y=[1.25,2.75,-2.75,-1.25]
        r=optimal_face(x,y,.25);self.check(r)
        self.assertEqual([v['coefficients']for v in r['vertices']],[['0','7/4','-1/2'],['7/4','0','-1/2']]);self.assertEqual(r['dimension'],1)
        self.assertEqual(r['always_nonzero'],[2]);self.assertEqual(r['always_zero'],[])
        for k in range(17):
            w=F(k,16);beta=[w*F(7,4),(1-w)*F(7,4),F(-1,2)]
            prediction=[sum(F(a)*b for a,b in zip(row,beta))for row in x]
            self.assertEqual(prediction,list(map(F,r['prediction'])))
            self.assertEqual(sum((F(a)-b)**2 for a,b in zip(y,prediction))/8+F(1,4)*sum(abs(b)for b in beta),F(5,8))

    def test_nonduplicate_dependent_columns_have_complete_segment(self):
        r=optimal_face([[1,0,.5],[0,1,.5]],[1.5,1.5],.25);self.check(r)
        self.assertEqual([v['coefficients']for v in r['vertices']],[['0','0','2'],['1','1','0']]);self.assertEqual(r['dimension'],1)
        self.assertEqual(r['vertex_average'],['1/2','1/2','1']);self.assertEqual(r['always_nonzero'],[])
        self.assertEqual(r['equicorrelation'],[0,1,2])

    def test_opposite_duplicates_and_higher_dimensional_simplex(self):
        for p in range(2,6):
            x=[[(-1)**j for j in range(p)]];r=optimal_face(x,[2],.25);self.check(r)
            self.assertEqual(r['dimension'],p-1);self.assertEqual(r['vertex_count'],p)
            for j,e in enumerate(r['coefficient_ranges']):
                self.assertEqual((e['lower'],e['upper']),('0','7/4')if j%2==0 else ('-7/4','0'))
            self.assertEqual(r['always_nonzero'],[])

    def test_equicorrelation_does_not_imply_possible_activity(self):
        x=[[1,1],[1,-1]];r=optimal_face(x,[1.25,.75],.25);self.check(r)
        self.assertEqual(r['equicorrelation'],[0,1]);self.assertEqual(r['always_zero'],[1]);self.assertEqual(r['dimension'],0)
        self.assertEqual(r['vertices'][0]['coefficients'],['3/4','0'])
        s=screen(x,[1.25,.75],[.75,0],.25)
        self.assertFalse(s['features'][1]['safe_zero'])

    def test_seeded_all_vertices_match_independent_gram_sign_enumeration(self):
        rng=random.Random(627)
        for case in range(30):
            p=1+case%4;n=3+case%4;x=[[rng.randrange(-2,3)for _ in range(p)]for _ in range(n)]
            if case%3==0 and p>1:
                for row in x:row[1]=row[0]
            y=[rng.randrange(-3,4)for _ in range(n)];lam=.25 if case%2 else .5
            r=optimal_face(x,y,lam);self.check(r);oracle=enumerate_optima(x,y,lam)
            self.assertEqual({tuple(v['coefficients'])for v in r['vertices']},{tuple(v['coefficients'])for v in oracle['witnesses']})
            for entry in r['subset_witnesses']:
                if entry['status']=='feasible_vertex':self.assertTrue(all(F(v)>=0 for v in entry['solution']))

    def test_zero_singleton_and_capacity(self):
        for x,y in [([[0,0]],[1]),([[1,0],[1,0]],[0,0]),([[1]],[.25])]:
            r=optimal_face(x,y,.25);self.check(r);self.assertTrue(r['unique']);self.assertEqual(r['dimension'],0)
        r=optimal_face([[1]*5 for _ in range(32)],[2]*32,.25);self.check(r);self.assertEqual(r['vertex_count'],5);self.assertEqual(r['dimension'],4)
        self.assertEqual(len(r['subset_witnesses']),32)

    def test_rectangular_solver_checks_all_rows_and_rank(self):
        self.assertEqual(_rectangular([[F(1),F(0),F(1)],[F(0),F(1),F(1)]],[F(2),F(3),F(5)]),[2,3])
        self.assertIsNone(_rectangular([[F(1),F(0),F(1)],[F(0),F(1),F(1)]],[F(2),F(3),F(6)]))
        self.assertIsNone(_rectangular([[F(1),F(2)],[F(2),F(4)]],[F(1),F(2)]))
        self.assertEqual(_rectangular([], [F(0),F(0)]),[]);self.assertIsNone(_rectangular([], [F(0),F(1)]))
        self.assertEqual(_rank([[F(1),F(2)],[F(2),F(4)]]),1);self.assertEqual(_rank([]),0)

    def test_row_column_sign_replication_and_response_scaling(self):
        x=[[1,0,.5],[0,1,.5]];y=[1.5,1.5];r=optimal_face(x,y,.25)
        self.assertEqual([v['coefficients']for v in optimal_face(x*3,y*3,.25)['vertices']],[v['coefficients']for v in r['vertices']])
        self.assertEqual(optimal_face(x[::-1],y[::-1],.25)['coefficient_ranges'],r['coefficient_ranges'])
        s=optimal_face(x,[-v for v in y],.25)
        self.assertEqual({tuple(F(v)for v in z['coefficients'])for z in s['vertices']},{tuple(-F(v)for v in z['coefficients'])for z in r['vertices']})
        s=optimal_face(x,[2*v for v in y],.5)
        self.assertEqual({tuple(F(v)for v in z['coefficients'])for z in s['vertices']},{tuple(2*F(v)for v in z['coefficients'])for z in r['vertices']})
        permutation=[2,0,1];s=optimal_face([[row[j]for j in permutation]for row in x],y,.25)
        self.assertEqual({tuple(z['coefficients'])for z in s['vertices']},{tuple(z['coefficients'][j]for j in permutation)for z in r['vertices']})

    def test_floating_solver_orders_stay_within_exact_face_ranges(self):
        x=[[1,1,1],[1,1,-1],[-1,-1,1],[-1,-1,-1]];y=[1.25,2.75,-2.75,-1.25];r=optimal_face(x,y,.25)
        for order in [[0,1,2],[2,1,0],[1,0,2]]:
            f=fit(x,y,.25,order=order);self.assertTrue(f['converged'])
            for b,e in zip(f['coefficients'],r['coefficient_ranges']):
                self.assertGreaterEqual(b,float(F(e['lower']))-1e-10);self.assertLessEqual(b,float(F(e['upper']))+1e-10)

    def test_strict_contracts_and_isolation(self):
        x=[[1,1],[1,1]];y=[2,2];before=copy.deepcopy((x,y));state=random.getstate();r=optimal_face(x,y,.25)
        self.assertEqual(before,(x,y));self.assertEqual(random.getstate(),state);r['vertices'][0]['coefficients'][0]='100';self.assertNotEqual(r,optimal_face(x,y,.25))
        for penalty in [True,0,-1,float('nan'),float('inf'),'1/4']:
            with self.assertRaises(ValueError):optimal_face(x,y,penalty)
        for a,b in [([[1]*6],[1]),([[1]]*33,[1]*33),([[1],[2,3]],[1,1]),([[1]],[True])]:
            with self.assertRaises(ValueError):optimal_face(a,b,.25)

    def test_cli_full_report_and_strict_errors(self):
        request={'schema_version':1,'X':[[1,0,.5],[0,1,.5]],'y':[1.5,1.5],'penalty':.25}
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'input.json'
            def run(raw):
                path.write_text(raw);return subprocess.run([sys.executable,'-m','sparselab','face',str(path)],capture_output=True,text=True)
            p=run(json.dumps(request));self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),optimal_face(request['X'],request['y'],.25))
            for raw in [json.dumps({**request,'extra':1}),json.dumps({**request,'penalty':True}),json.dumps({**request,'schema_version':True}),json.dumps(request)[:-1]+',"schema_version":1}','{"x":NaN}','['*2000,' '*100001]:
                p=run(raw);self.assertEqual(p.returncode,2,p.stderr);self.assertEqual(p.stdout,'')

    def test_full_dyadic_grid_iff_face_membership_not_just_vertices(self):
        x=[[1,0,.5],[0,1,.5]];y=[F(3,2)]*2;r=optimal_face(x,list(map(float,y)),.25)
        minimum=F(r['dual']);grid=[F(i,4)for i in range(-2,11)];count=0
        for beta in product(grid,repeat=3):
            prediction=[beta[0]+beta[2]/2,beta[1]+beta[2]/2]
            objective=sum((a-b)**2 for a,b in zip(y,prediction))/4+sum(abs(b)for b in beta)/4
            member=(0<=beta[2]<=2 and beta[0]==beta[1]==1-beta[2]/2)
            self.assertEqual(objective==minimum,member)
            self.assertGreaterEqual(objective,minimum)
            count+=member
        self.assertEqual(count,5)
