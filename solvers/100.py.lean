import «.formalsw».solvers.«100.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p100.py»

namespace solvers.«100»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.ArrangedProbability
set_option Elab.async false
private abbrev program := Solvers.P100Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)

namespace next_solution
theorem result (u v : Int) (heap : Mutable.Heap) :
    ReturnsValue program "next_solution" [number u, number v] heap (.ref heap.length)
      (heap ++ [.tuple [number (3*u+4*v), number (2*u+3*v)]]) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply Evaluates.tuple (nonempty := by simp)
  refine .cons (middle := heap) ?_ (.cons (middle := heap) ?_ .nil)
  all_goals
    repeat first
      | apply Evaluates.binary (middle := heap) (afterRight := heap)
      | exact Evaluates.constant
      | exact Evaluates.lookup rfl
      | rfl
end next_solution

namespace blue_red_for_total_limit
private def body : List Stmt :=
  [.assign [.name "n"] (.binOp (.binOp (.name "u") .add (.constant (.int 1))) .floorDiv (.constant (.int 2))),
   .assign [.name "b"] (.binOp (.binOp (.name "v") .add (.constant (.int 1))) .floorDiv (.constant (.int 2))),
   .ifThen (.compare (.name "n") [(.gt, .name "limit")])
     [.assign [.name "r"] (.binOp (.name "n") .sub (.name "b")),
      .return (.tuple [.name "b", .name "r", .name "n"])] [],
   .assign [.tuple [.name "u", .name "v"]] (.call (.name "next_solution") [.name "u", .name "v"])]
private def loop : List Stmt := [.whileDo (.constant (.bool true)) body []]
private def resultTuple (pair : Int × Int) : Mutable.Object :=
  .tuple [number pair.1, number (pair.2-pair.1), number pair.2]

/-- The Pell coordinates are twice the total and blue counts minus one. -/
private theorem loop_result (limit b t : Int) (positiveB : 0 < b) (positiveT : 1 < t)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getU : env "u" = some (number (2*t-1))) (getV : env "v" = some (number (2*b-1)))
    (getLimit : env "limit" = some (number limit)) :
    ∃ finalHeap, BlockReturns program env loop heap (.ref finalHeap.length)
      (finalHeap ++ [resultTuple (firstOver limit b t positiveB positiveT)]) := by
  let withN := setLocal env "n" (number t)
  let ready := setLocal withN "b" (number b)
  have totalValue : Evaluates program env
      (.binOp (.binOp (.name "u") .add (.constant (.int 1))) .floorDiv (.constant (.int 2)))
      heap (number t) heap := by
    apply Evaluates.binary (av := number (2*t-1+1)) (bv := number 2) (middle := heap) (afterRight := heap)
    · exact Evaluates.binary (Evaluates.lookup getU) Evaluates.constant rfl
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary, number]
  have blueValue : Evaluates program withN
      (.binOp (.binOp (.name "v") .add (.constant (.int 1))) .floorDiv (.constant (.int 2)))
      heap (number b) heap := by
    apply Evaluates.binary (av := number (2*b-1+1)) (bv := number 2) (middle := heap) (afterRight := heap)
    · exact Evaluates.binary (Evaluates.lookup (by simpa [withN, setLocal] using getV)) Evaluates.constant rfl
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary, number]
  by_cases done : limit < t
  · refine ⟨heap, ?_⟩
    rw [firstOver, if_pos done]
    apply BlockReturns.while_loop Evaluates.constant rfl
    apply BlockReturns.assign totalValue
    apply BlockReturns.assign blueValue
    apply BlockReturns.branch (value := .scalar (.bool true)) (flag := true) (middle := heap)
    · exact Evaluates.compare (av := number t) (bv := number limit) (middle := heap) (afterRight := heap)
        (Evaluates.lookup (by simp [withN, setLocal]))
        (Evaluates.lookup (by simpa [ready, withN, setLocal] using getLimit))
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
    · rfl
    · apply BlockReturns.assign (value := number (t-b)) (middle := heap)
      · exact Evaluates.binary (av := number t) (bv := number b) (middle := heap) (afterRight := heap)
          (Evaluates.lookup (by simp [withN, setLocal])) (Evaluates.lookup rfl) rfl
      apply BlockReturns.return_value
      exact Evaluates.tuple (.cons (Evaluates.lookup (by simp [withN, setLocal]))
        (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup (by simp [withN, setLocal])) .nil))) (by simp)
  · let nextU := 3*(2*t-1)+4*(2*b-1)
    let nextV := 2*(2*t-1)+3*(2*b-1)
    let nextEnv := assignNames ready ["u", "v"] [number nextU, number nextV]
    let nextHeap := heap ++ [.tuple [number nextU, number nextV]]
    obtain ⟨finalHeap, rest⟩ := loop_result limit (3*b+2*t-2) (4*b+3*t-3) (by omega) (by omega)
      nextHeap nextEnv
      (by simp [nextEnv, assignNames, setLocal, number, nextU]; omega)
      (by simp [nextEnv, assignNames, setLocal, number, nextV]; omega)
      (by simpa [nextEnv, ready, withN, assignNames, setLocal] using getLimit)
    refine ⟨finalHeap, ?_⟩
    rw [firstOver, if_neg done]
    apply BlockReturns.while_loop Evaluates.constant rfl
    apply BlockReturns.assign totalValue
    apply BlockReturns.assign blueValue
    apply BlockReturns.branch (value := .scalar (.bool false)) (flag := false) (middle := heap)
    · exact Evaluates.compare (av := number t) (bv := number limit) (middle := heap) (afterRight := heap)
        (Evaluates.lookup (by simp [withN, setLocal]))
        (Evaluates.lookup (by simpa [ready, withN, setLocal] using getLimit))
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
    · rfl
    · apply BlockReturns.unpack (items := [number nextU, number nextV])
        (value := .ref heap.length) (middle := nextHeap) (afterUnpack := nextHeap) rfl
      · exact Evaluates.call (.cons (Evaluates.lookup (by simpa [ready, withN, setLocal] using getU))
          (.cons (Evaluates.lookup (by simpa [ready, withN, setLocal] using getV)) .nil))
          (next_solution.result (2*t-1) (2*b-1) heap)
      · simp [Mutable.sequence, nextHeap]
      · rfl
      · simpa [body, loop, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest
termination_by (limit + 1 - t).toNat
decreasing_by omega

theorem result (limit : Int) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "blue_red_for_total_limit" [number limit] heap (.ref finalHeap.length)
      (finalHeap ++ [resultTuple (firstOver limit 3 4 (by decide) (by decide))]) := by
  let entry := functionEnv ((findFunctionIn program.body "blue_red_for_total_limit").getD ([], [])).1
    ((findFunctionIn program.body "blue_red_for_total_limit").getD ([], [])).2 [number limit] []
  obtain ⟨finalHeap, rest⟩ := loop_result limit 3 4 (by decide) (by decide)
    (heap ++ [.tuple [number 7, number 5]]) (assignNames entry ["u", "v"] [number 7, number 5])
    (by simp [assignNames, setLocal, number]) (by simp [assignNames, setLocal, number]) rfl
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.unpack (items := [number 7, number 5])
    (afterUnpack := heap ++ [.tuple [number 7, number 5]]) rfl
  · exact Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant .nil)) (by simp)
  · simp [Mutable.sequence]
  · rfl
  · exact rest
end blue_red_for_total_limit

namespace solve
theorem result (limit : Int) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [number limit] heap (number (answer limit)) finalHeap := by
  obtain ⟨finalHeap, run⟩ := blue_red_for_total_limit.result limit heap
  refine ⟨finalHeap ++ [blue_red_for_total_limit.resultTuple (firstOver limit 3 4 (by decide) (by decide))], ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply Evaluates.subscript (by intros; simp)
    (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) run) Evaluates.constant
  simp [Mutable.getItem, Mutable.index, blue_red_for_total_limit.resultTuple, answer, number, Except.map]

theorem «Returns» (limit : Int) : (HeapCalls.scalarFunction program "solve" [.int limit]).Returns := by
  obtain ⟨_, run⟩ := result limit []
  exact run.scalar_returns (values := [.int limit])

theorem «Equiv(naive)» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (HeapCalls.scalarFunction P100Ast.module "naive" [.int limit]) := by
  obtain ⟨_, left⟩ := result limit []
  obtain ⟨_, right⟩ := data.«project-euler-statements».data.py.project_euler_statements.p100.naive.result limit []
  have optimized := left.scalar_finishes (values := [.int limit])
  have reference := right.scalar_finishes (values := [.int limit])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem «Supported» :
    (∀ limit : Int, (HeapCalls.scalarFunction program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, (HeapCalls.scalarFunction P100Ast.module "naive" [.int limit]).Supported) :=
  ⟨fun limit => («Returns» limit).supported,
    data.«project-euler-statements».data.py.project_euler_statements.p100.naive.«Supported»⟩
end solve
end solvers.«100»
