import ast, copy, io, json, unittest
from pathlib import Path
import numpy as np
from tao.printing import TreePrinter
from tao.portable import predict_margin


def linear_form(node):
    """Independent symbolic reader for printed affine expressions."""
    if isinstance(node,ast.Expression):return linear_form(node.body)
    if isinstance(node,ast.Constant):return {None:float(node.value)}
    if isinstance(node,ast.Name):return {node.id:1.}
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return {k:-v for k,v in linear_form(node.operand).items()}
    if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub)):
        a,b=linear_form(node.left),linear_form(node.right);sign=-1 if isinstance(node.op,ast.Sub) else 1
        for k,v in b.items():a[k]=a.get(k,0)+sign*v
        return a
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult):
        a,b=linear_form(node.left),linear_form(node.right)
        if set(a)!={None}:raise AssertionError('Expected scalar coefficient')
        return {k:a[None]*v for k,v in b.items()}
    raise AssertionError('Unexpected expression in tree printout')


class TreePrintingTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads((Path(__file__).resolve().parents[1]/'fpga_handoff/v3/compact_refit_3t_d4/standardized/model.json').read_text())

    def test_full_precision_actual_printout_recovers_all_coefficients_and_predictions(self):
        before=copy.deepcopy(self.spec);printer=TreePrinter(self.spec);expressions={}
        for line in printer.format_forest().splitlines():
            if '(x) = ' in line:
                name,text=line.strip().split('(x) = ');expressions[name]=linear_form(ast.parse(text,mode='eval'))
        count=reachable=0
        for t in range(3):
            records=printer.records(t);self.assertEqual(len(records),31)
            for r in records:
                key=('h' if r['kind']=='split' else 'f')+f'_{t}_{r["index"]}'
                form=expressions[key]
                for name,value in zip(self.spec['features'],r['weights']):self.assertEqual(form.get(name,0),value)
                self.assertEqual(form.get(None,0),r['bias'])
                n=sum(v!=0 for v in r['weights'])+(r['bias']!=0);count+=n
                if r['reachable']:reachable+=n
        self.assertEqual(count,418);self.assertEqual(reachable,379);self.assertEqual(self.spec,before)
        x=np.random.default_rng(420).normal(size=(80,16));margin=[];routes=[]
        for row in x:
            env=dict(zip(self.spec['features'],row));selected=[];score=0.
            def value(name):return sum(v*(1 if k is None else env[k]) for k,v in expressions[name].items())
            for t in range(3):
                node=0
                while node<15:node=2*node+1+(value(f'h_{t}_{node}')>0)
                leaf=node-15;selected.append(leaf);score-=value(f'f_{t}_{leaf}')
            routes.append(selected);margin.append(score)
        expected,leaves,_=predict_margin(self.spec,x)
        np.testing.assert_array_equal(routes,leaves);np.testing.assert_allclose(margin,expected,atol=1e-10,rtol=1e-10)

    def test_stream_method_and_reachability_use_algebra_not_empirical_rows(self):
        s=copy.deepcopy(self.spec);s['trees'][0]['split_weights'][0]=[0.]*16;s['trees'][0]['split_bias'][0]=0.
        p=TreePrinter(s);out=io.StringIO();p.print_tree(0,file=out)
        self.assertEqual(out.getvalue(),p.format_tree(0));self.assertIn('CONSTANT: always LEFT',out.getvalue())
        records=p.records(0);self.assertTrue(next(r for r in records if r['path']=='L')['reachable'])
        self.assertFalse(next(r for r in records if r['path']=='R')['reachable'])
        self.assertIn('UNREACHABLE',out.getvalue());self.assertIn('f_0_15(x)',out.getvalue())

    def test_tiny_terms_are_not_pruned_and_constant_leaves_are_supported(self):
        p=TreePrinter(self.spec);w=[0.]*16;w[0]=1e-25
        self.assertIn('e-25',p.affine(w,0));self.assertEqual(p.affine([0.]*16,0),'0')
        s=copy.deepcopy(self.spec);s['leaf_mode']='constant'
        for tree in s['trees']:tree['leaf_parameters']=[[.25] for _ in range(16)]
        q=TreePrinter(s);self.assertIn('f_0_0(x) = 0.25',q.format_tree(0))

    def test_invalid_precision_topology_and_tree_index_are_rejected(self):
        for n in (0,18,True,2.5):
            with self.assertRaises(ValueError):TreePrinter(self.spec,n)
        for n in (-1,3,True):
            with self.assertRaises(ValueError):TreePrinter(self.spec).format_tree(n)
        s=copy.deepcopy(self.spec);s['trees'][0]['split_weights'][0].pop()
        with self.assertRaises(ValueError):TreePrinter(s)
