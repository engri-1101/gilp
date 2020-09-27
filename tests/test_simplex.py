import pytest
import mock
import numpy as np
import gilp.simplex as sm


class TestLP:

    def test_init_exceptions(self):
        with pytest.raises(ValueError, match='.*b should have shape .*'):
            A = np.array([[1,0],[0,1]])
            b = np.array([[1],[2],[3]])
            c = np.array([[1],[2]])
            sm.LP(A,b,c)
        with pytest.raises(ValueError, match='.*c should have shape .*'):
            A = np.array([[1,0],[0,1]])
            b = np.array([[1],[2]])
            c = np.array([[1],[2],[3]])
            sm.LP(A,b,c)

    @pytest.mark.parametrize("lp,n,m,A,b,c,equality",[
        (sm.LP(np.array([[1,2],[3,0]]),
               np.array([3,4]),
               np.array([1,2])),
         2,2,
         np.array([[1,2],[3,0]]),
         np.array([[3],[4]]),
         np.array([[1],[2]]),
         False),
        (sm.LP(np.array([[1,2,3],[3,0,1]]),
               np.array([[3],[4]]),
               np.array([[1],[2],[3]]),
               equality=True),
         3,2,
         np.array([[1,2,3],[3,0,1]]),
         np.array([[3],[4]]),
         np.array([[1],[2],[3]]),
         True)])
    def test_init(self,lp,n,m,A,b,c,equality):
        actual = lp.get_coefficients()
        assert n == actual[0]
        assert m == actual[1]
        assert (A == actual[2]).all()
        assert (b == actual[3]).all()
        assert (c == actual[4]).all()
        assert (equality == lp.equality)

    def test_get_bfs(self, degenerate_lp):
        lp = degenerate_lp
        bfs = np.array([[2],[4],[0],[0],[4],[1],[0]])
        assert (bfs == lp.get_basic_feasible_sol([0,1,4,5,6])).all()
        assert (bfs == lp.get_basic_feasible_sol([0,1,2,4,5])).all()
        assert (bfs == lp.get_basic_feasible_sol([0,1,3,4,5])).all()
        with pytest.raises(sm.InvalidBasis):
            lp.get_basic_feasible_sol([1,2,3,4])
        with pytest.raises(sm.InvalidBasis):
            lp.get_basic_feasible_sol([0,1,2,4,5,6])
        with pytest.raises(sm.InfeasibleBasicSolution):
            lp.get_basic_feasible_sol([0,1,2,3,5])

    def test_get_all_bfs(self):
        A = np.array([[1,1],[-2,1]])
        b = np.array([[6],[0]])
        c = np.array([[1],[1]])
        lp = sm.LP(A,b,c)
        bfs = [np.array([[2],[4],[0],[0]]),
               np.array([[0],[0],[6],[0]]),
               np.array([[6],[0],[0],[12]]),
               np.array([[0],[0],[6],[0]]),
               np.array([[0],[0],[6],[0]])]
        bases = [[0,1],[0,2],[0,3],
                 [1,2],[2,3]]
        values = [6,0,6,0,0]
        actual = lp.get_basic_feasible_solns()

        assert all(np.allclose(x,y,atol=1e-7) for x,y in zip(bfs, actual[0]))
        assert bases == actual[1]
        assert values == actual[2]

    def test_tableau(self, degenerate_lp):
        T = np.array([[1,0,0,1,1,0,0,0,10],
                      [0,1,0,1,-1,0,0,0,2],
                      [0,0,1,0,1,0,0,0,4],
                      [0,0,0,-1,2,1,0,0,4],
                      [0,0,0,-1,1,0,1,0,1],
                      [0,0,0,2,-3,0,0,1,0]])
        assert (T == degenerate_lp.get_tableau([0,1,4,5,6])).all()
        with pytest.raises(sm.InvalidBasis):
            degenerate_lp.get_tableau([1,2,3])


class TestSimplexIteration:

    def test_bad_inputs(self, klee_minty_3d_lp):
        with pytest.raises(ValueError,match='Invalid pivot rule.*'):
            sm.simplex_iteration(lp=klee_minty_3d_lp,
                                 x=np.array([[5],[5],[0],[0],[0],[65]]),
                                 B=[0,1,5],
                                 pivot_rule='invalid')
        with pytest.raises(ValueError,match='x should have shape.*'):
            sm.simplex_iteration(lp=klee_minty_3d_lp,
                                 x=np.array([[5],[5],[0],[0],[0]]),
                                 B=[0,1,5],
                                 pivot_rule='bland')
        with pytest.raises(ValueError,match='.*different basic feasible.*'):
            sm.simplex_iteration(lp=klee_minty_3d_lp,
                                 x=np.array([[5],[5],[0],[0],[0],[65]]),
                                 B=[0,1,2],
                                 pivot_rule='bland')

    def test_bland(self, klee_minty_3d_lp):
        actual = sm.simplex_iteration(lp=klee_minty_3d_lp,
                                      x=np.array([[5],[5],[0],[0],[0],[65]]),
                                      B=[0,1,5],
                                      pivot_rule='bland')
        assert (np.array([[5],[5],[65],[0],[0],[0]]) == actual[0]).all()
        actual[1].sort()
        assert [0,1,2] == actual[1]
        assert 95 == actual[2]
        assert not actual[3]

    def test_min_index(self, klee_minty_3d_lp):
        actual = sm.simplex_iteration(lp=klee_minty_3d_lp,
                                      x=np.array([[5],[5],[0],[0],[0],[65]]),
                                      B=[0,1,5],
                                      pivot_rule='min_index')
        assert (np.array([[5],[5],[65],[0],[0],[0]]) == actual[0]).all()
        actual[1].sort()
        assert [0,1,2] == actual[1]
        assert 95 == actual[2]
        assert not actual[3]

    def test_dantzig(self, klee_minty_3d_lp):
        actual = sm.simplex_iteration(lp=klee_minty_3d_lp,
                                      x=np.array([[0],[0],[0],[5],[25],[125]]),
                                      B=[3,4,5],
                                      pivot_rule='dantzig')
        assert (np.array([[5],[0],[0],[0],[5],[85]]) == actual[0]).all()
        actual[1].sort()
        assert [0,4,5] == actual[1]
        assert 20 == actual[2]
        assert not actual[3]

    def test_max_reduced_cost(self, klee_minty_3d_lp):
        actual = sm.simplex_iteration(lp=klee_minty_3d_lp,
                                      x=np.array([[0],[0],[0],[5],[25],[125]]),
                                      B=[3,4,5],
                                      pivot_rule='max_reduced_cost')
        assert (np.array([[5],[0],[0],[0],[5],[85]]) == actual[0]).all()
        actual[1].sort()
        assert [0,4,5] == actual[1]
        assert 20 == actual[2]
        assert not actual[3]

    def test_greatest_ascent1(self, klee_minty_3d_lp):
        actual = sm.simplex_iteration(lp=klee_minty_3d_lp,
                                      x=np.array([[0],[0],[0],[5],[25],[125]]),
                                      B=[3,4,5],
                                      pivot_rule='greatest_ascent')
        assert (np.array([[0],[0],[125],[5],[25],[0]]) == actual[0]).all()
        actual[1].sort()
        assert [2,3,4] == actual[1]
        assert 125 == actual[2]
        assert not actual[3]

    def test_greatest_ascent2(self, klee_minty_3d_lp):
        actual = sm.simplex_iteration(lp=klee_minty_3d_lp,
                                      x=np.array([[0],[0],[125],[5],[25],[0]]),
                                      B=[2,3,4],
                                      pivot_rule='greatest_ascent')
        assert (np.array([[0],[0],[125],[5],[25],[0]]) == actual[0]).all()
        actual[1].sort()
        assert [2,3,4] == actual[1]
        assert 125 == actual[2]
        assert actual[3]

    def test_manual_select(self, klee_minty_3d_lp):
        with mock.patch('builtins.input', return_value="2"):
            actual = sm.simplex_iteration(lp=klee_minty_3d_lp,
                                          x=np.array([[0],[0],[0],
                                                      [5],[25],[125]]),
                                          B=[3,4,5],
                                          pivot_rule='manual_select')
            assert (np.array([[0],[25],[0],[5],[0],[25]]) == actual[0]).all()
            actual[1].sort()
            assert [1,3,5] == actual[1]
            assert 50 == actual[2]
            assert not actual[3]


class TestSimplex():

    def test_bad_inputs(self, klee_minty_3d_lp, unbounded_lp):
        with pytest.raises(ValueError,match='Invalid pivot rule.*'):
            sm.simplex(lp=klee_minty_3d_lp,
                       pivot_rule='invalid')
        with pytest.raises(ValueError,match='.*should have shape.*'):
            sm.simplex(lp=klee_minty_3d_lp,
                       initial_solution=np.array([[5],[5],[0],[0]]))
        with pytest.raises(ValueError,match='Iteration limit*'):
            sm.simplex(lp=klee_minty_3d_lp,
                       iteration_limit=-1)
        with pytest.raises(sm.UnboundedLinearProgram):
            # Make sure the initial solution is ignored and no error is raised
            sm.simplex(unbounded_lp,
                       initial_solution=np.array([[2],[2],[0],[0]]))
        with pytest.raises(sm.UnboundedLinearProgram):
            sm.simplex(unbounded_lp,'greatest_ascent')

    def test_simplex(self, klee_minty_3d_lp):
        actual = sm.simplex(klee_minty_3d_lp,
                            pivot_rule='dantzig',
                            initial_solution=np.array([[0],[0],[0],
                                                       [5],[25],[125]]))
        bfs = np.array([[0],[0],[125],[5],[25],[0]])
        bases = [2,3,4]
        assert np.allclose(bfs,actual[0],atol=1e-7)
        assert np.allclose(bases,actual[1],atol=1e-7)
        assert 125 == actual[2]
        assert actual[3]

    def test_initial_solution(self, klee_minty_3d_lp):
        actual = sm.simplex(klee_minty_3d_lp,
                            initial_solution=np.array([[5],[5],[65],
                                                       [0],[0],[0]]),
                            pivot_rule='dantzig')
        bfs = np.array([[0],[0],[125],[5],[25],[0]])
        bases = [2,3,4]
        assert np.allclose(bfs,actual[0],atol=1e-7)
        assert np.allclose(bases,actual[1],atol=1e-7)
        assert 125 == actual[2]
        assert actual[3]

    def test_degenerate_init_sol(self):
        A = np.array([[1,1],[0,1],[1,-1],[1,0],[-2,1]])
        b = np.array([[6],[4],[2],[3],[0]])
        c = np.array([[1],[0]])
        lp = sm.LP(A,b,c)
        sm.simplex(lp,initial_solution=np.array([[2],[4],[0],[0],[0],[1],[0]]))

    def test_iteration_limit(self, klee_minty_3d_lp):
        actual = sm.simplex(klee_minty_3d_lp,
                            pivot_rule='dantzig',
                            iteration_limit=3,
                            initial_solution=np.array([[0],[0],[0],
                                                       [5],[25],[125]]))
        bfs = np.array([[0],[25],[0],[5],[0],[25]])
        bases = [1,5,3]
        assert np.allclose(bfs,actual[0],atol=1e-7)
        assert np.allclose(bases,actual[1],atol=1e-7)
        assert 50 == actual[2]
        assert not actual[3]


class TestPhaseOne():

    @pytest.mark.parametrize("lp,bfs",[
        (sm.LP(np.array([[1,1,0],[-1,1,-1]]),
               np.array([[3],[1]]),
               np.array([[2],[1],[0]]),
               equality=True), (np.array([[1],[2],[0]]),[0,1]))])
    def test_phase_one(self,lp,bfs):
        x,B = sm.phase_one(lp)
        assert all(x == bfs[0])
        assert B == bfs[1]

    @pytest.mark.parametrize("lp",[
        (sm.LP(np.array([[1],[-1]]),
               np.array([2,-3]),
               np.array([1]))),
        (sm.LP(np.array([[1,0],[0,-1],[-1,0]]),
               np.array([2,-3,-3]),
               np.array([1,1]))),
        (sm.LP(np.array([[0,1],[0,-1],[-1,0]]),
               np.array([2,-3,-3]),
               np.array([1,2])))])
    def test_infeasible(self,lp):
        with pytest.raises(sm.Infeasible):
            sm.phase_one(lp)

    @pytest.mark.parametrize("lp,bfs",[
        (sm.LP(np.array([[1],[1]]),
               np.array([[0],[0]]),
               np.array([[1]]),
               equality=True), (np.array([[0]]),[0])),
        (sm.LP(np.array([[1],[1]]),
               np.array([[3],[3]]),
               np.array([[1]]),
               equality=True), (np.array([[3]]),[0])),
        (sm.LP(np.array([[1,1],[1,1]]),
               np.array([[1],[1]]),
               np.array([[2],[1]]),
               equality=True), (np.array([[1],[0]]),[0])),
        (sm.LP(np.array([[1,1],[1,0]]),
               np.array([[1],[1]]),
               np.array([[1],[1]]),
               equality=True), (np.array([[1],[0]]),[0,1]))])
    def test_degenerate(self,lp,bfs):
        x,B = sm.phase_one(lp)
        assert all(x == bfs[0])
        assert B == bfs[1]


@pytest.mark.parametrize("A,t",[
    (np.array([[1,0],[0,1]]), True),
    (np.array([[0,1],[1,0]]), True),
    (np.array([[1,0,0],[0,1,0]]), False),
    (np.array([[2,0,0],[0,0,3],[0,1,0]]), True)])
def test_invertible(A,t):
    assert sm.invertible(A) == t


@pytest.mark.parametrize("lp,x,val",[
    (sm.LP(np.array([[-2,2],[2,2]]),
           np.array([[1],[7]]),
           np.array([[1],[2]])),
     np.array([[2],[1]]),
     4.0),
    (sm.LP(np.array([[1,1],[5,9]]),
           np.array([[6],[45]]),
           np.array([[5],[8]])),
     np.array([[0],[5]]),
     40.0),
    (sm.LP(np.array([[1,10],[1,0]]),
           np.array([[20],[2]]),
           np.array([[0],[5]])),
     np.array([[0],[2]]),
     10.0),
    (sm.LP(np.array([[-2,2,1,0],[2,2,0,1]]),
           np.array([[1],[7]]),
           np.array([[1],[2],[0],[0]]),equality=True),
     np.array([[2],[1],[3],[1]]),
     4.0),
    (sm.LP(np.array([[1,1,1,0],[5,9,0,1]]),
           np.array([[6],[45]]),
           np.array([[5],[8],[0],[0]]),equality=True),
     np.array([[0],[5],[1],[0]]),
     40.0),
    (sm.LP(np.array([[1,10,1,0],[1,0,0,1]]),
           np.array([[20],[2]]),
           np.array([[0],[5],[0],[0]]),equality=True),
     np.array([[0],[2],[0],[2]]),
     10.0)])
def test_branch_and_bound(lp,x,val):
    ans = sm.branch_and_bound(lp)
    assert all(x == ans[0])
    assert val == ans[1]
