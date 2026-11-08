import solvers.«84.py.lean.d».LandingChanceBranch
import ProjectEulerSolutions.Lib.Monopoly

namespace solvers.«84».resolve_landing.chance
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

private theorem membership (square : Int) (values : List Int) (nonempty : values ≠ [])
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getSquare : env "square" = some (.scalar (.int square))) :
    HeapMemoCalls.Evaluates program pool.spec env
      (.membership (.name "square") (.tuple (values.map fun value => .constant (.int value))) false) heap
      (.scalar (.bool (decide (square ∈ values))))
      (heap ++ [.tuple (values.map fun value => .scalar (.int value))]) globals := by
  have arguments : HeapCalls.Evaluates.Arguments program env (values.map fun value => .constant (.int value)) heap
      (values.map fun value => .scalar (.int value)) heap globals := by
    clear nonempty
    induction values with
    | nil => exact .nil
    | cons value values ih => exact .cons HeapCalls.Evaluates.constant ih
  apply HeapMemoCalls.Evaluates.from_heap (HeapCalls.Extends.refl program) pool.disjoint
    (preserved := TuplePool.preserves_append pool.catalog heap _)
  apply HeapCalls.Evaluates.membership (.cons (HeapCalls.Evaluates.lookup getSquare) (.cons
    (HeapCalls.Evaluates.tuple arguments (by simpa using nonempty)) .nil))
  exact HeapCalls.membership_tuple_ints _ heap.length square values (by simp) false

/-- Every Chance square executes to the exact aggregated uniform-card weights. -/
theorem result (square : Int) (member : square ∈ [7, 22, 36])
    (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ReturnsRel program pool.spec "resolve_landing" [.scalar (.int square)] heap (Post heap square) globals := by
  let env := HeapCalls.functionEnv pool.definition.1 pool.definition.2 [.scalar (.int square)] globals
  have getSquare : env "square" = some (.scalar (.int square)) := by cases globals <;> rfl
  have notJail : square ≠ 30 := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    omega
  have notCommunity : square ∉ [2, 17, 33] := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member ⊢
    omega
  have jailTest : HeapMemoCalls.Evaluates program pool.spec env (.compare (.name "square") [(.eq, .constant (.int 30))])
      heap (.scalar (.bool false)) heap globals := by
    apply HeapMemoCalls.Evaluates.compare (HeapMemoCalls.Evaluates.lookup getSquare) HeapMemoCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, notJail]
  let cc := Mutable.Object.tuple [.scalar (.int 2), .scalar (.int 17), .scalar (.int 33)]
  let ch := Mutable.Object.tuple [.scalar (.int 7), .scalar (.int 22), .scalar (.int 36)]
  have communityTest : HeapMemoCalls.Evaluates program pool.spec env
      (.membership (.name "square") (.tuple [.constant (.int 2), .constant (.int 17), .constant (.int 33)]) false)
      heap (.scalar (.bool false)) (heap ++ [cc]) globals := by
    simpa [notCommunity] using membership square [2, 17, 33] (by simp) env heap globals getSquare
  have chanceTest : HeapMemoCalls.Evaluates program pool.spec env
      (.membership (.name "square") (.tuple [.constant (.int 7), .constant (.int 22), .constant (.int 36)]) false)
      (heap ++ [cc]) (.scalar (.bool true)) ((heap ++ [cc]) ++ [ch]) globals := by
    simpa [member] using membership square [7, 22, 36] (by simp) env (heap ++ [cc]) globals getSquare
  apply HeapMemoCalls.BlockReturnsRel.function (args := pool.definition.1) (body := pool.definition.2) (by decide) rfl rfl
  apply HeapMemoCalls.BlockReturnsRel.branch jailTest.rel
  rintro value after ⟨rfl, same⟩
  subst after
  refine ⟨false, rfl, ?_⟩
  apply HeapMemoCalls.BlockReturnsRel.branch communityTest.rel
  rintro value after ⟨rfl, same⟩
  subst after
  refine ⟨false, rfl, ?_⟩
  apply HeapMemoCalls.BlockReturnsRel.branch chanceTest.rel
  rintro value after ⟨rfl, same⟩
  subst after
  refine ⟨true, rfl, ?_⟩
  apply (branch_result square env ((heap ++ [cc]) ++ [ch]) globals getSquare _).consequence
  exact fun _ _ post => post.prefix.prefix

theorem data_model (square : Nat) (member : square ∈ [7, 22, 36]) :
    data square = (Monopoly.landingWeights square).map (fun entry =>
      (((entry.1.1 : Int), entry.1.2), (entry.2 : Int))) := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl <;> rfl

end solvers.«84».resolve_landing.chance
