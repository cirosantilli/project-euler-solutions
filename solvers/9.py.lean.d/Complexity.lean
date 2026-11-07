import solvers.«9.py.lean.d».Source
import «.formalsw».solvers.«9.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p9.py.lean.d».Source
import lang.py.FormalSW.HeapCallsCost
import lang.py.FormalSW.ScalarSortCost
import lang.py.FormalSW.WeightedExecutionCost

/-! Counted execution of the original Euler 9 source ASTs. Integer arithmetic
is atomic. Transition bounds alone do not bound collection traversal work. -/
namespace solvers.«9».complexity
open FormalSW.PyAst
open FormalSW.PyAst.HeapCalls.Cost
set_option maxRecDepth 4096
set_option maxHeartbeats 800000

namespace reference
private abbrev program := P9Ast.module
open data.«project-euler-statements».data.py.project_euler_statements.p9

private theorem ordered_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (a b c : Int)
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) :
    ∃ cost, cost ≤ 9 ∧ EvaluatesIn program env ordered heap cost
      (.scalar (.bool (decide (a < b ∧ b < c)))) heap := by
  by_cases ab : a < b
  · refine ⟨9, by omega, ?_⟩
    apply EvaluatesIn.chain (cb := 6) (EvaluatesIn.lookup ha)
    apply EvaluatesIn.ComparesIn.next (cb := 3) (compared := .scalar (.bool true))
      (afterComparison := heap) (EvaluatesIn.lookup hb)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, ab]
    · rfl
    · apply EvaluatesIn.ComparesIn.last (EvaluatesIn.lookup hc)
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, ab]
  · refine ⟨6, by omega, ?_⟩
    apply EvaluatesIn.chain (cb := 3) (EvaluatesIn.lookup ha)
    apply EvaluatesIn.ComparesIn.stop (EvaluatesIn.lookup hb)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, ab]
    · simp [Mutable.truth, scalarTruth, ab]

private theorem square_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (name : String) (value : Int)
    (found : env name = some (.scalar (.int value))) :
    EvaluatesIn program env (square name) heap 5 (.scalar (.int (value * value))) heap :=
  EvaluatesIn.binary (EvaluatesIn.lookup found) (EvaluatesIn.lookup found) rfl

private theorem condition_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (n a b : Int)
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int (n - a - b)))) :
    ∃ cost, cost ≤ 33 ∧ EvaluatesIn program env condition heap cost
      (.scalar (.bool (decide (valid n a b)))) heap := by
  have squares : EvaluatesIn program env pythagorean heap 21
      (.scalar (.bool (decide (a * a + b * b = (n - a - b) * (n - a - b))))) heap := by
    apply EvaluatesIn.compare
      (EvaluatesIn.binary (square_cost env heap "a" a ha) (square_cost env heap "b" b hb) rfl)
      (square_cost env heap "c" (n - a - b) hc)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  obtain ⟨cost, bounded, orderRun⟩ := ordered_cost env heap a b (n - a - b) ha hb hc
  by_cases order : a < b ∧ b < n - a - b
  · refine ⟨cost + 24, by omega, ?_⟩
    apply EvaluatesIn.boolean_next (cb := 22) (value := .scalar (.bool true)) (flag := true) (middle := heap)
    · simpa [order] using orderRun
    · rfl
    · rfl
    · apply EvaluatesIn.boolean_last (flag := decide (valid n a b))
      · simpa [valid, order.1, order.2] using squares
      · rfl
  · have invalid : ¬valid n a b := fun h => order ⟨h.1, h.2.1⟩
    refine ⟨cost + 2, by omega, ?_⟩
    apply EvaluatesIn.boolean_stop (flag := false)
    · simpa [invalid, order] using orderRun
    · simp [Mutable.truth, scalarTruth, invalid]
    · rfl

/-- Every actual reference loop body takes at most 62 transitions, including
the set method call when a triplet is found. Set traversal is not yet weighted. -/
theorem innerBody_cost (env : HeapCalls.Env) (heap : Mutable.Heap)
    (items : List Int) (n a b : Int)
    (hn : env "n" = some (.scalar (.int n))) (ha : env "a" = some (.scalar (.int a)))
    (hb : env "b" = some (.scalar (.int b))) (hout : env "out" = some (.ref heap.length)) :
    ∃ cost, cost ≤ 62 ∧ BlockContinuesIn program env innerBody (heap ++ [.set (ints items)]) cost
      (HeapCalls.setLocal env "c" (.scalar (.int (n - a - b))))
      (heap ++ [.set (ints (if valid n a b then insertInt items (a * b * (n - a - b)) else items))]) := by
  have assigned : BlockContinuesIn program env [innerBody[0]] (heap ++ [.set (ints items)]) 11
      (HeapCalls.setLocal env "c" (.scalar (.int (n - a - b)))) (heap ++ [.set (ints items)]) :=
    BlockContinuesIn.assign
    (EvaluatesIn.binary
      (EvaluatesIn.binary (EvaluatesIn.lookup hn) (EvaluatesIn.lookup ha) rfl)
      (EvaluatesIn.lookup hb) rfl)
  let env' := HeapCalls.setLocal env "c" (.scalar (.int (n - a - b)))
  have getA : env' "a" = some (.scalar (.int a)) := by simpa [env', HeapCalls.setLocal] using ha
  have getB : env' "b" = some (.scalar (.int b)) := by simpa [env', HeapCalls.setLocal] using hb
  have getC : env' "c" = some (.scalar (.int (n - a - b))) := rfl
  have getOut : env' "out" = some (.ref heap.length) := by simpa [env', HeapCalls.setLocal] using hout
  obtain ⟨guardCost, bounded, guard⟩ := condition_cost env' (heap ++ [.set (ints items)]) n a b getA getB getC
  by_cases good : valid n a b
  · refine ⟨11 + (guardCost + 16 + 2), by omega, ?_⟩
    apply BlockContinuesIn.append assigned
    apply BlockContinuesIn.branch (flag := true) (value := .scalar (.bool true))
    · simpa [good] using guard
    · rfl
    · apply BlockContinuesIn.discard (value := .scalar .none)
      have found : (heap ++ [.set (ints items)])[heap.length]? = some (.set (ints items)) := by simp
      have prod : EvaluatesIn program env' product (heap ++ [.set (ints items)]) 9
          (.scalar (.int (a * b * (n - a - b)))) (heap ++ [.set (ints items)]) :=
        EvaluatesIn.binary
          (EvaluatesIn.binary (EvaluatesIn.lookup getA) (EvaluatesIn.lookup getB) rfl)
          (EvaluatesIn.lookup getC) rfl
      have inserted := EvaluatesIn.add (EvaluatesIn.lookup getOut) found prod found
      simpa [good, ints, ScalarSet.insert_ints, insertInt] using inserted
  · refine ⟨11 + (guardCost + 0 + 2), by omega, ?_⟩
    apply BlockContinuesIn.append assigned
    apply BlockContinuesIn.branch (cc := guardCost) (cb := 0) (flag := false) (value := .scalar (.bool false))
    · simpa [good] using guard
    · rfl
    · simpa [good] using (BlockContinuesIn.nil (module := program) (env := env')
        (heap := heap ++ [.set (ints items)]))
private theorem row_cost (count : Nat) (n a b : Int) (bound : n + 1 = b + count)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (items : List Int)
    (hn : env "n" = some (.scalar (.int n))) (ha : env "a" = some (.scalar (.int a)))
    (hout : env "out" = some (.ref heap.length)) :
    ∃ finalEnv cost, count + 1 ≤ cost ∧ cost ≤ 63 * count + 1 ∧ BlockContinuesIn program env
      [.rangeStepLoop "b" b (n + 1) 1 innerBody []] (heap ++ [.set (ints items)]) cost
      finalEnv (heap ++ [.set (ints (collectRow n a count b items))]) ∧
      finalEnv "n" = some (.scalar (.int n)) ∧ finalEnv "a" = some (.scalar (.int a)) ∧
      finalEnv "out" = some (.ref heap.length) := by
  induction count generalizing b env items with
  | zero =>
      refine ⟨env, 1, by omega, by omega, ?_, hn, ha, hout⟩
      apply BlockContinuesIn.range_done
      · simp [Calls.rangeContinues]; omega
      · exact BlockContinuesIn.nil
  | succ count ih =>
      let envB := HeapCalls.setLocal env "b" (.scalar (.int b))
      let envC := HeapCalls.setLocal envB "c" (.scalar (.int (n - a - b)))
      let updated := if valid n a b then insertInt items (a * b * (n - a - b)) else items
      have getN : envB "n" = some (.scalar (.int n)) := by simpa [envB, HeapCalls.setLocal] using hn
      have getA : envB "a" = some (.scalar (.int a)) := by simpa [envB, HeapCalls.setLocal] using ha
      have getOut : envB "out" = some (.ref heap.length) := by simpa [envB, HeapCalls.setLocal] using hout
      obtain ⟨iterationCost, iterationBound, iteration⟩ := innerBody_cost envB heap items n a b getN getA rfl getOut
      obtain ⟨finalEnv, nextCost, nextLower, nextBound, next, finalN, finalA, finalOut⟩ := ih (b + 1) (by omega) envC updated
        (by simpa [envC, HeapCalls.setLocal] using getN)
        (by simpa [envC, HeapCalls.setLocal] using getA)
        (by simpa [envC, HeapCalls.setLocal] using getOut)
      refine ⟨finalEnv, iterationCost + nextCost + 1, by omega, by omega, ?_, finalN, finalA, finalOut⟩
      apply BlockContinuesIn.range_step
      · simp [Calls.rangeContinues]; omega
      · simpa only [innerBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          collectRow] using iteration.append next

private theorem rows_cost (n count : Nat) (a : Int) (bound : (n : Int) + 1 = a + count)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (items : List Int)
    (hn : env "n" = some (.scalar (.int n))) (hout : env "out" = some (.ref heap.length)) :
    ∃ finalEnv cost, (n + 1) * count + 1 ≤ cost ∧ cost ≤ (63 * n + 12) * count + 1 ∧ BlockContinuesIn program env
      [.rangeStepLoop "a" a (n + 1) 1 outerBody []] (heap ++ [.set (ints items)]) cost
      finalEnv (heap ++ [.set (ints (collectRows n count a items))]) ∧
      finalEnv "n" = some (.scalar (.int n)) ∧ finalEnv "out" = some (.ref heap.length) := by
  induction count generalizing a env items with
  | zero =>
      refine ⟨env, 1, by omega, by omega, ?_, hn, hout⟩
      apply BlockContinuesIn.range_done
      · simp [Calls.rangeContinues]; omega
      · exact BlockContinuesIn.nil
  | succ count ih =>
      let envA := HeapCalls.setLocal env "a" (.scalar (.int a))
      have getN : envA "n" = some (.scalar (.int n)) := by simpa [envA, HeapCalls.setLocal] using hn
      have getOut : envA "out" = some (.ref heap.length) := by simpa [envA, HeapCalls.setLocal] using hout
      obtain ⟨rowEnv, rowCost, rowLower, rowBound, row, rowN, _, rowOut⟩ := row_cost n n a 1 (by omega) envA heap items getN rfl getOut
      have iteration : BlockContinuesIn program envA outerBody (heap ++ [.set (ints items)]) (rowCost + 10)
          rowEnv (heap ++ [.set (ints (collectRow n a n 1 items))]) := by
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
          BlockContinuesIn.for_range
            (.cons EvaluatesIn.constant (.cons
              (EvaluatesIn.binary (EvaluatesIn.lookup getN) EvaluatesIn.constant rfl) .nil))
            rfl row
      obtain ⟨finalEnv, nextCost, nextLower, nextBound, next, finalN, finalOut⟩ := ih (a + 1) (by omega) rowEnv
        (collectRow n a n 1 items) rowN rowOut
      refine ⟨finalEnv, (rowCost + 10) + nextCost + 1, by
        rw [Nat.mul_succ]; omega, by
        rw [Nat.mul_succ]; omega, ?_, finalN, finalOut⟩
      apply BlockContinuesIn.range_step
      · simp [Calls.rangeContinues]; omega
      · simpa only [outerBody, innerBody, Calls.loopContinuation, Calls.loopControl,
          Calls.loopControlStmt, collectRows] using iteration.append next

/-- The helper's two source ranges, including allocation, loop setup and return. -/
theorem triplet_products_cost (n : Nat) (heap : Mutable.Heap) :
    ∃ cost, n * n ≤ cost ∧ cost ≤ (63 * n + 12) * n + 18 ∧
      ReturnsIn program "triplet_products" [.scalar (.int n)] heap cost (.ref heap.length)
        (heap ++ [.set (ints (products n))]) := by
  let env := HeapCalls.setLocal
    (HeapCalls.argumentEnv [{name := "n", annotation := some (.name "int")}] [.scalar (.int n)])
    "out" (.ref heap.length)
  obtain ⟨finalEnv, loopCost, lower, bounded, run, _, finalOut⟩ := rows_cost n n 1 (by omega) env heap [] rfl rfl
  refine ⟨(2 + ((8 + ((loopCost + 2) + 1) + 1)) + 2) + 1, by
    simp only [Nat.add_mul, Nat.one_mul] at lower; omega, by omega, ?_⟩
  apply BlockReturnsIn.function rfl rfl
  apply BlockReturnsIn.assign (value := .ref heap.length) (middle := heap ++ [.set []])
  · exact EvaluatesIn.builtin .nil (by simp [HeapCalls.primitive, Mutable.allocateSet, Except.map])
  · apply BlockReturnsIn.for_range
      (.cons EvaluatesIn.constant (.cons
        (EvaluatesIn.binary (EvaluatesIn.lookup rfl) EvaluatesIn.constant rfl) .nil)) rfl
    exact run.returns (BlockReturnsIn.return_value (EvaluatesIn.lookup finalOut))

private theorem maximum_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat)
    (first : Int) (rest : List Int) (found : heap[address]? = some (.set (ints (first :: rest))))
    (lookup : env "vals" = some (.ref address)) :
    EvaluatesIn program env (.builtinCall .max [.name "vals"]) heap 4
      (.scalar (.int (rest.foldl max first))) heap := by
  apply EvaluatesIn.builtin (.cons (EvaluatesIn.lookup lookup) .nil)
  simp [HeapCalls.primitive, found, ints, ScalarSet.maximum, HeapCalls.scalarResult,
    ScalarSet.maximumFrom_ints, List.mapM_map, List.mapM_pure, Function.comp_def]

/-- Full reference call, including helper call and final max. This is a
transition bound; maximum and set insertion still need traversal weights. -/
theorem naive_cost (n : Nat) (heap : Mutable.Heap) :
    ∃ cost, n * n ≤ cost ∧ cost ≤ (63 * n + 12) * n + 32 ∧
      ReturnsIn program "naive" [.scalar (.int n)] heap cost
        (.scalar (.int (largest (products n)))) (heap ++ [.set (ints (products n))]) := by
  obtain ⟨helperCost, lower, bounded, helper⟩ := triplet_products_cost n heap
  have tail : ∃ cost, cost ≤ 8 ∧ BlockReturnsIn program
      (HeapCalls.setLocal (HeapCalls.argumentEnv [{name := "n", annotation := some (.name "int")}]
        [.scalar (.int n)]) "vals" (.ref heap.length))
      [.return (.ifExp (.name "vals") (.builtinCall .max [.name "vals"]) (.constant (.int 0)))]
      (heap ++ [.set (ints (products n))]) cost (.scalar (.int (largest (products n))))
      (heap ++ [.set (ints (products n))]) := by
    cases casesProducts : products n with
    | nil =>
        refine ⟨5, by omega, ?_⟩
        apply BlockReturnsIn.return_value
        apply EvaluatesIn.conditional (cb := 1) (flag := false) (EvaluatesIn.lookup rfl)
        · simp [Mutable.truth, ints]
        · exact EvaluatesIn.constant
    | cons first rest =>
        refine ⟨8, by omega, ?_⟩
        apply BlockReturnsIn.return_value
        apply EvaluatesIn.conditional (cb := 4) (flag := true) (EvaluatesIn.lookup rfl)
        · simp [Mutable.truth, ints]
        · exact maximum_cost _ _ heap.length first rest (by simp) rfl
  obtain ⟨tailCost, tailBound, tail⟩ := tail
  refine ⟨((2 + helperCost + 1) + tailCost + 2) + 1, by omega, by omega, ?_⟩
  apply BlockReturnsIn.function rfl rfl
  exact BlockReturnsIn.assign (EvaluatesIn.call (.cons (EvaluatesIn.lookup rfl) .nil) helper) tail

/-- A terminating counted trace of the original Python `naive` entry point. -/
theorem timeBound (n : Nat) :
    (HeapCalls.call P9Ast.module "naive" [.scalar (.int n)] []).execution.TimeBound
      ((63 * n + 12) * n + 33) := by
  obtain ⟨cost, _, bounded, run⟩ := naive_cost n []
  refine ⟨cost + 1, _, (run 0 []).finishes (.done rfl), by omega⟩

/-- Quadratic interpreter-transition complexity, with no source-shape classifier. -/
theorem isBigO : Execution.IsBigO
    (fun n : Nat => (HeapCalls.call P9Ast.module "naive" [.scalar (.int n)] []).execution)
    (fun _ => True) id (fun n => n * n) := by
  refine ⟨108, 1, ?_⟩
  intro n _ large
  apply (timeBound n).mono
  have h1 : n ≤ n * n := by simpa using Nat.mul_le_mul_left n large
  have h2 : 1 ≤ n * n := Nat.le_trans large h1
  calc
    (63 * n + 12) * n + 33 = 63 * (n * n) + 12 * n + 33 := by
      rw [Nat.add_mul, Nat.mul_assoc]
    _ ≤ 63 * (n * n) + 12 * (n * n) + 33 * (n * n) := by omega
    _ = 108 * (n * n) := by omega

/-- Any successful or exceptional counted execution has at least T² transitions.
Determinism connects this lower bound to the actual entry point. -/
theorem lowerBound (n : Nat)
    (run : FinishesIn (HeapCalls.step P9Ast.module 0)
      ⟨.operation (.call "naive") [.scalar (.int n)], [], []⟩ cost outcome) :
    n * n ≤ cost := by
  obtain ⟨witness, lower, _, trace⟩ := naive_cost n []
  have exactTrace := (trace 0 []).finishes (FinishesIn.done rfl)
  have equal := (run.unique exactTrace).1
  omega

/-- The reference cannot have a linear bound even in the cheaper transition model. -/
theorem not_linear : ¬Execution.IsBigO
    (fun n : Nat => (HeapCalls.call P9Ast.module "naive" [.scalar (.int n)] []).execution)
    (fun _ => True) id id := by
  rintro ⟨constant, threshold, bounded⟩
  let n := threshold + constant + 1
  obtain ⟨cost, outcome, run, upper⟩ := bounded n trivial (by dsimp [n]; omega)
  have lower := lowerBound n run
  have comparison : constant * n < n * n := Nat.mul_lt_mul_of_pos_right (by dsimp [n]; omega) (by dsimp [n]; omega)
  change cost ≤ constant * n at upper
  omega

/-- Charging at least one unit for every transition preserves the source's
quadratic lower bound, including models that also charge set scans. -/
theorem weighted_lowerBound (n : Nat) (charge : HeapCalls.State → Nat)
    (positive : ∀ state, 1 ≤ charge state)
    (run : FinishesWith (HeapCalls.step P9Ast.module 0) charge
      ⟨.operation (.call "naive") [.scalar (.int n)], [], []⟩ cost outcome) :
    n * n ≤ cost := by
  have steps {state : HeapCalls.State} {cost : Nat} {outcome : Except ExceptionObject Mutable.Completion}
      (h : FinishesWith (HeapCalls.step P9Ast.module 0) charge state cost outcome) :
      ∃ transitions, FinishesIn (HeapCalls.step P9Ast.module 0) state transitions outcome ∧ transitions ≤ cost := by
    induction h with
    | done terminal => exact ⟨1, .done terminal, positive _⟩
    | @next state following cost outcome transition rest ih =>
        obtain ⟨transitions, executed, bounded⟩ := ih
        exact ⟨transitions + 1, .next transition executed, by have := positive state; omega⟩
  obtain ⟨transitions, executed, bounded⟩ := steps run
  exact Nat.le_trans (lowerBound n executed) bounded

/-- The returned set has at most one product per first leg. -/
theorem products_length (n : Nat) : (products n).length ≤ n :=
  data.«project-euler-statements».data.py.project_euler_statements.p9.products_length n

end reference
namespace optimized
open solvers.«9».runtimeSource
private abbrev program := Solvers.P9Ast.module

/-- Execute the actual tuple display and sorted call, retaining both allocations. -/
theorem sortedExpression_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (a b c : Int)
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) (ac : a ≤ c) (bc : b ≤ c) :
    EvaluatesIn program env sortedExpression heap 11 (.ref (heap.length + 1))
      (sortedHeap heap a b c) := by
  have tupleRun := EvaluatesIn.tuple (globals := []) (module := program) (heap := heap)
    (.cons (EvaluatesIn.lookup ha) (.cons (EvaluatesIn.lookup hb)
      (.cons (EvaluatesIn.lookup hc) .nil))) (by simp)
  apply EvaluatesIn.builtin (.cons tupleRun .nil)
  have operation := HeapCalls.primitive_sorted_tuple (heap ++ [.tuple (triple a b c)])
    heap.length [.int a, .int b, .int c] [.int (min a b), .int (max a b), .int c]
    (by simp [triple]) (ScalarSort.three_top a b c ac bc)
  simpa [sortedHeap, triple, List.append_assoc] using operation

theorem sortedAssignment_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (a b c : Int)
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) (ac : a ≤ c) (bc : b ≤ c) :
    BlockContinuesIn program env
      [.assign [.tuple [.name "a", .name "b", .name "c"]] sortedExpression] heap 13
      (HeapCalls.assignNames env ["a", "b", "c"] (triple (min a b) (max a b) c))
      (sortedHeap heap a b c) := by
  apply BlockContinuesIn.unpack rfl (sortedExpression_cost env heap a b c ha hb hc ac bc)
  · simp [Mutable.sequence, sortedHeap, triple]
  · rfl

/-- Locate the sorting assignment in the checked AST rather than postulating
an independent loop body. Failure to match the source makes the next theorem fail. -/
def sourceSortedStatement : Option Stmt := do
  let (_, body) ← findFunctionIn Solvers.P9Ast.module.body "solve"
  let .whileDo _ outer _ ← (executableBody body)[2]? | none
  let .forIn _ _ inner _ ← outer[0]? | none
  inner[6]?

theorem source_sorted : sourceSortedStatement =
    some (.assign [.tuple [.name "a", .name "b", .name "c"]] sortedExpression) := rfl

/-- The actual sorted expression takes eleven transitions, retains two
three-element allocations, and invokes at most three scalar comparisons. -/
theorem sorted_fixed3 (env : HeapCalls.Env) (heap : Mutable.Heap) (a b c : Int)
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) (ac : a ≤ c) (bc : b ≤ c) :
    EvaluatesIn program env sortedExpression heap 11 (.ref (heap.length + 1)) (sortedHeap heap a b c) ∧
    (sortedHeap heap a b c).length = heap.length + 2 ∧
    ∃ comparisons, comparisons ≤ 3 ∧
      ScalarSort.Cost.Sorts [.int a, .int b, .int c] comparisons [.int (min a b), .int (max a b), .int c] := by
  exact ⟨sortedExpression_cost env heap a b c ha hb hc ac bc,
    by simp [sortedHeap], ScalarSort.Cost.three_top a b c ac bc⟩


/-- Seventeen transitions for the actual outer-loop guard. -/
theorem outerCondition_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total m : Int)
    (ht : env "total" = some (.scalar (.int total))) (hm : env "m" = some (.scalar (.int m))) :
    EvaluatesIn program env outerCondition heap 17
      (.scalar (.bool (decide (2 * m * (m + 1) ≤ total)))) heap := by
  apply EvaluatesIn.compare
    (EvaluatesIn.binary
      (EvaluatesIn.binary EvaluatesIn.constant (EvaluatesIn.lookup hm) rfl)
      (EvaluatesIn.binary (EvaluatesIn.lookup hm) EvaluatesIn.constant rfl) rfl)
    (EvaluatesIn.lookup ht)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem source_outer_condition :
    ((findFunctionIn program.body "solve").bind fun (_, body) =>
      (executableBody body)[2]?.bind fun
        | .whileDo condition _ _ => some condition
        | _ => none) = some outerCondition := rfl


private theorem triangleCondition_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total a b c : Int)
    (ht : env "total" = some (.scalar (.int total)))
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) :
    ∃ flag cost, cost ≤ 43 ∧ EvaluatesIn program env triangleCondition heap cost (.scalar (.bool flag)) heap := by
  have first : EvaluatesIn program env (.compare (.name "a") [(.gt, .constant (.int 0))])
      heap 5 (.scalar (.bool (decide (0 < a)))) heap := by
    apply EvaluatesIn.compare (EvaluatesIn.lookup ha) EvaluatesIn.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  have second : EvaluatesIn program env
      (.compare (.binOp (square "a") .add (square "b")) [(.eq, square "c")])
      heap 21 (.scalar (.bool (decide (a * a + b * b = c * c)))) heap := by
    apply EvaluatesIn.compare
      (EvaluatesIn.binary
        (EvaluatesIn.binary (EvaluatesIn.lookup ha) (EvaluatesIn.lookup ha) rfl)
        (EvaluatesIn.binary (EvaluatesIn.lookup hb) (EvaluatesIn.lookup hb) rfl) rfl)
      (EvaluatesIn.binary (EvaluatesIn.lookup hc) (EvaluatesIn.lookup hc) rfl)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  have third : EvaluatesIn program env
      (.compare (.binOp (.binOp (.name "a") .add (.name "b")) .add (.name "c")) [(.eq, .name "total")])
      heap 13 (.scalar (.bool (decide (a + b + c = total)))) heap := by
    apply EvaluatesIn.compare
      (EvaluatesIn.binary
        (EvaluatesIn.binary (EvaluatesIn.lookup ha) (EvaluatesIn.lookup hb) rfl)
        (EvaluatesIn.lookup hc) rfl) (EvaluatesIn.lookup ht)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases positive : 0 < a
  · have firstTrue : EvaluatesIn program env (.compare (.name "a") [(.gt, .constant (.int 0))]) heap 5 (.scalar (.bool true)) heap := by
      simpa [positive] using first
    by_cases squares : a * a + b * b = c * c
    · refine ⟨decide (a + b + c = total), 43, by omega, ?_⟩
      apply EvaluatesIn.boolean_next (cb := 36) (flag := true) firstTrue rfl rfl
      apply EvaluatesIn.boolean_next (ca := 21) (cb := 14) (flag := true)
      · simpa [squares] using second
      · rfl
      · rfl
      · exact EvaluatesIn.boolean_last (flag := decide (a + b + c = total)) third rfl
    · refine ⟨false, 29, by omega, ?_⟩
      apply EvaluatesIn.boolean_next (cb := 22) (flag := true) firstTrue rfl rfl
      apply EvaluatesIn.boolean_stop (cost := 21) (flag := false)
      · simpa [squares] using second
      · rfl
      · rfl
  · refine ⟨false, 7, by omega, ?_⟩
    apply EvaluatesIn.boolean_stop (cost := 5) (flag := false)
    · simpa [positive] using first
    · rfl
    · rfl

private theorem updateBest_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total a b c best : Int)
    (ht : env "total" = some (.scalar (.int total)))
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) (hbest : env "best" = some (.scalar (.int best))) :
    ∃ newBest finalEnv cost, cost ≤ 61 ∧ BlockContinuesIn program env updateBest heap cost finalEnv heap ∧
      finalEnv "total" = some (.scalar (.int total)) ∧ finalEnv "m" = env "m" ∧
      finalEnv "best" = some (.scalar (.int newBest)) := by
  obtain ⟨flag, cc, bounded, condition⟩ := triangleCondition_cost env heap total a b c ht ha hb hc
  cases flag with
  | false =>
      refine ⟨best, env, cc + 0 + 2, by omega, ?_, ht, rfl, hbest⟩
      exact BlockContinuesIn.branch condition rfl BlockContinuesIn.nil
  | true =>
      have maximum : EvaluatesIn program env (.builtinCall .max [.name "best", productExpression]) heap 14
          (.scalar (.int (max best (a * b * c)))) heap := by
        apply EvaluatesIn.builtin
          (.cons (EvaluatesIn.lookup hbest) (.cons
            (EvaluatesIn.binary
              (EvaluatesIn.binary (EvaluatesIn.lookup ha) (EvaluatesIn.lookup hb) rfl)
              (EvaluatesIn.lookup hc) rfl) .nil))
        by_cases larger : best < a * b * c
        · simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom,
            HeapCalls.scalarResult, scalarCompare, evalCmpOp, larger, Int.max_def]
          omega
        · simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom,
            HeapCalls.scalarResult, scalarCompare, evalCmpOp, larger, Int.max_def]
          omega
      refine ⟨max best (a * b * c), HeapCalls.setLocal env "best" (.scalar (.int (max best (a * b * c)))),
        cc + 16 + 2, by omega, BlockContinuesIn.branch condition rfl (BlockContinuesIn.assign maximum),
        ?_, ?_, rfl⟩
      · simpa [HeapCalls.setLocal] using ht
      · simp [HeapCalls.setLocal]

/-- Sorting arbitrary integer triples preserves arity, including unsuccessful
candidates; no mathematical validity assumption is needed for the cost proof. -/
private theorem sorted_any_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (a b c : Int)
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) :
    ∃ x y z, BlockContinuesIn program env
      [.assign [.tuple [.name "a", .name "b", .name "c"]] sortedExpression] heap 13
      (HeapCalls.assignNames env ["a", "b", "c"] (triple x y z))
      (heap ++ [.tuple (triple a b c), .list (triple x y z)]) := by
  have length : (ScalarSort.ints [a, b, c]).length = 3 := (ScalarSort.ints_perm [a, b, c]).length_eq
  obtain ⟨x, y, z, values⟩ : ∃ x y z, ScalarSort.ints [a, b, c] = [x, y, z] := by
    generalize ScalarSort.ints [a, b, c] = xs at length ⊢
    match xs with
    | [] | [_] | [_, _] => simp at length
    | x :: y :: z :: rest =>
        have empty : rest = [] := by simpa using length
        exact ⟨x, y, z, by rw [empty]⟩
  refine ⟨x, y, z, ?_⟩
  have tupleRun := EvaluatesIn.tuple (globals := []) (module := program) (heap := heap)
    (.cons (EvaluatesIn.lookup ha) (.cons (EvaluatesIn.lookup hb)
      (.cons (EvaluatesIn.lookup hc) .nil))) (by simp)
  have sorted : EvaluatesIn program env sortedExpression heap 11 (.ref (heap.length + 1))
      (heap ++ [.tuple (triple a b c), .list (triple x y z)]) := by
    apply EvaluatesIn.builtin (.cons tupleRun .nil)
    have operation := HeapCalls.primitive_sorted_tuple (heap ++ [.tuple (triple a b c)])
      heap.length [.int a, .int b, .int c] [.int x, .int y, .int z]
      (by simp [triple]) (by simpa [values] using ScalarSort.sort_ints [a, b, c])
    simpa [triple, List.append_assoc] using operation
  apply BlockContinuesIn.unpack rfl sorted
  · simp [Mutable.sequence, triple]
  · rfl


private theorem coefficientA_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (k m n : Int)
    (hk : env "k" = some (.scalar (.int k)))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n))) :
    EvaluatesIn program env aExpression heap 17 (.scalar (.int (k * (m * m - n * n)))) heap :=
  EvaluatesIn.binary (EvaluatesIn.lookup hk)
    (EvaluatesIn.binary
      (EvaluatesIn.binary (EvaluatesIn.lookup hm) (EvaluatesIn.lookup hm) rfl)
      (EvaluatesIn.binary (EvaluatesIn.lookup hn) (EvaluatesIn.lookup hn) rfl) rfl) rfl

private theorem coefficientB_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (k m n : Int)
    (hk : env "k" = some (.scalar (.int k)))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n))) :
    EvaluatesIn program env bExpression heap 13 (.scalar (.int (k * (2 * m * n)))) heap :=
  EvaluatesIn.binary (EvaluatesIn.lookup hk)
    (EvaluatesIn.binary
      (EvaluatesIn.binary EvaluatesIn.constant (EvaluatesIn.lookup hm) rfl)
      (EvaluatesIn.lookup hn) rfl) rfl

private theorem coefficientC_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (k m n : Int)
    (hk : env "k" = some (.scalar (.int k)))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n))) :
    EvaluatesIn program env cExpression heap 17 (.scalar (.int (k * (m * m + n * n)))) heap :=
  EvaluatesIn.binary (EvaluatesIn.lookup hk)
    (EvaluatesIn.binary
      (EvaluatesIn.binary (EvaluatesIn.lookup hm) (EvaluatesIn.lookup hm) rfl)
      (EvaluatesIn.binary (EvaluatesIn.lookup hn) (EvaluatesIn.lookup hn) rfl) rfl) rfl

private theorem coefficientBody_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total m n s : Int)
    (ht : env "total" = some (.scalar (.int total)))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n)))
    (hs : env "s" = some (.scalar (.int s))) (positive : 0 < s) :
    BlockContinuesIn program env coefficientBody heap 60 (coefficientEnv env total m n s) heap := by
  have divide : EvaluatesIn program env kExpression heap 5 (.scalar (.int (total / s))) heap := by
    apply EvaluatesIn.binary (EvaluatesIn.lookup ht) (EvaluatesIn.lookup hs)
    simp [Mutable.binary, scalarBinary, positive, (show s ≠ 0 by omega)]
  apply BlockContinuesIn.append (cb := 53) (BlockContinuesIn.assign divide)
  apply BlockContinuesIn.append (cb := 34) (BlockContinuesIn.assign
    (coefficientA_cost _ heap (total / s) m n rfl
      (by simpa [HeapCalls.setLocal] using hm) (by simpa [HeapCalls.setLocal] using hn)))
  apply BlockContinuesIn.append (cb := 19) (BlockContinuesIn.assign
    (coefficientB_cost _ heap (total / s) m n rfl
      (by simpa [HeapCalls.setLocal] using hm) (by simpa [HeapCalls.setLocal] using hn)))
  exact BlockContinuesIn.assign (coefficientC_cost _ heap (total / s) m n rfl
    (by simpa [HeapCalls.setLocal] using hm) (by simpa [HeapCalls.setLocal] using hn))


private theorem successBody_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total m n s best : Int)
    (ht : env "total" = some (.scalar (.int total)))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n)))
    (hs : env "s" = some (.scalar (.int s))) (hbest : env "best" = some (.scalar (.int best)))
    (positive : 0 < s) :
    ∃ newBest finalEnv finalHeap cost, cost ≤ 134 ∧
      BlockContinuesIn program env successBody heap cost finalEnv finalHeap ∧
      finalEnv "total" = some (.scalar (.int total)) ∧ finalEnv "m" = some (.scalar (.int m)) ∧
      finalEnv "best" = some (.scalar (.int newBest)) := by
  have coefficients := coefficientBody_cost env heap total m n s ht hm hn hs positive
  obtain ⟨x, y, z, sorted⟩ := sorted_any_cost (coefficientEnv env total m n s) heap
    (total / s * (m * m - n * n)) (total / s * (2 * m * n)) (total / s * (m * m + n * n)) rfl rfl rfl
  let sortedEnv := HeapCalls.assignNames (coefficientEnv env total m n s) ["a", "b", "c"] (triple x y z)
  have keptT : sortedEnv "total" = some (.scalar (.int total)) := by
    simpa [sortedEnv, coefficientEnv, HeapCalls.assignNames, triple, HeapCalls.setLocal] using ht
  have keptM : sortedEnv "m" = some (.scalar (.int m)) := by
    simpa [sortedEnv, coefficientEnv, HeapCalls.assignNames, triple, HeapCalls.setLocal] using hm
  have keptBest : sortedEnv "best" = some (.scalar (.int best)) := by
    simpa [sortedEnv, coefficientEnv, HeapCalls.assignNames, triple, HeapCalls.setLocal] using hbest
  obtain ⟨newBest, finalEnv, cc, bounded, updated, finalT, finalM, finalBest⟩ :=
    updateBest_cost sortedEnv _ total x y z best keptT rfl rfl rfl keptBest
  refine ⟨newBest, finalEnv, _, (60 + 13) + cc, by omega, (coefficients.append sorted).append updated,
    finalT, ?_, finalBest⟩
  exact finalM.trans keptM


private theorem sExpression_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (m n : Int)
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n))) :
    EvaluatesIn program env sExpression heap 13 (.scalar (.int (2 * m * (m + n)))) heap :=
  EvaluatesIn.binary
    (EvaluatesIn.binary EvaluatesIn.constant (EvaluatesIn.lookup hm) rfl)
    (EvaluatesIn.binary (EvaluatesIn.lookup hm) (EvaluatesIn.lookup hn) rfl) rfl

private theorem skipCondition_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total s : Int)
    (ht : env "total" = some (.scalar (.int total)))
    (hs : env "s" = some (.scalar (.int s))) (positive : 0 < s) :
    EvaluatesIn program env skipCondition heap 9 (.scalar (.bool (decide (total % s ≠ 0)))) heap := by
  have remainder : EvaluatesIn program env (.binOp (.name "total") .mod (.name "s"))
      heap 5 (.scalar (.int (total % s))) heap := by
    apply EvaluatesIn.binary (EvaluatesIn.lookup ht) (EvaluatesIn.lookup hs)
    simp [Mutable.binary, scalarBinary, positive, (show s ≠ 0 by omega)]
  apply EvaluatesIn.compare remainder EvaluatesIn.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]
/-- Count the original range body, including both `continue` and sorting paths. -/
theorem innerRange_cost (total best : Int) (m n : Nat) (positive : 0 < n)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (ht : env "total" = some (.scalar (.int total))) (hm : env "m" = some (.scalar (.int m)))
    (hbest : env "best" = some (.scalar (.int best))) :
    ∃ newBest finalEnv finalHeap cost, cost ≤ 162 * (m - n) + 1 ∧
      BlockContinuesIn program env [.rangeStepLoop "n" n m 1 innerBody []] heap cost finalEnv finalHeap ∧
      finalEnv "total" = some (.scalar (.int total)) ∧ finalEnv "m" = some (.scalar (.int m)) ∧
      finalEnv "best" = some (.scalar (.int newBest)) := by
  by_cases before : n < m
  · let denominator : Int := 2 * (m : Int) * ((m : Int) + n)
    let nextEnv := HeapCalls.setLocal env "n" (.scalar (.int n))
    let withS := HeapCalls.setLocal nextEnv "s" (.scalar (.int denominator))
    have nt : nextEnv "total" = some (.scalar (.int total)) := by simpa [nextEnv, HeapCalls.setLocal] using ht
    have nm : nextEnv "m" = some (.scalar (.int m)) := by simpa [nextEnv, HeapCalls.setLocal] using hm
    have st : withS "total" = some (.scalar (.int total)) := by simpa [withS, HeapCalls.setLocal] using nt
    have sm : withS "m" = some (.scalar (.int m)) := by simpa [withS, HeapCalls.setLocal] using nm
    have sbest : withS "best" = some (.scalar (.int best)) := by simpa [withS, nextEnv, HeapCalls.setLocal] using hbest
    have denominatorPositive : 0 < denominator :=
      Int.mul_pos (Int.mul_pos (by decide) (by omega)) (by omega)
    have assigned := BlockContinuesIn.assign (name := "s") (sExpression_cost nextEnv heap m n nm rfl)
    have guard := skipCondition_cost withS heap total denominator st rfl denominatorPositive
    have running : Calls.rangeContinues (n : Int) m 1 = true := by simp [Calls.rangeContinues, before]
    by_cases divides : total % denominator = 0
    · obtain ⟨rowBest, rowEnv, rowHeap, successCost, successBound, success, rowT, rowM, rowB⟩ :=
        successBody_cost withS heap total m n denominator best st sm rfl rfl sbest denominatorPositive
      obtain ⟨newBest, finalEnv, finalHeap, restCost, restBound, rest, keptT, keptM, keptB⟩ :=
        innerRange_cost total rowBest m (n + 1) (by omega) rowEnv rowHeap rowT rowM rowB
      refine ⟨newBest, finalEnv, finalHeap,
        (15 + (9 + ((successCost + (restCost + 1)) + 1) + 1)) + 1, by omega, ?_, keptT, keptM, keptB⟩
      apply BlockContinuesIn.range_step running
      apply BlockContinuesIn.append assigned
      have continuation := success.append (BlockContinuesIn.loop_resume rest)
      have falseGuard : EvaluatesIn program withS skipCondition heap 9 (.scalar (.bool false)) heap := by
        simpa [divides] using guard
      intro following anchors stack
      exact .next rfl ((falseGuard anchors _).trans (.next rfl (by
        simpa only [Int.natCast_add, Int.natCast_one] using continuation following anchors stack)))
    · obtain ⟨newBest, finalEnv, finalHeap, restCost, restBound, rest, keptT, keptM, keptB⟩ :=
        innerRange_cost total best m (n + 1) (by omega) withS heap st sm sbest
      refine ⟨newBest, finalEnv, finalHeap,
        (15 + (9 + ((restCost + 1) + 1) + 1)) + 1, by omega, ?_, keptT, keptM, keptB⟩
      apply BlockContinuesIn.range_step running
      apply BlockContinuesIn.append assigned
      have trueGuard : EvaluatesIn program withS skipCondition heap 9 (.scalar (.bool true)) heap := by
        simpa [divides] using guard
      intro following anchors stack
      refine .next rfl ((trueGuard anchors _).trans (.next rfl ?_))
      refine ReachesIn.next (following :=
        ⟨.block (.rangeStepLoop "n" ((n : Int) + 1) m 1 innerBody [] :: following) withS, stack, heap⟩) rfl ?_
      simpa only [Int.natCast_add, Int.natCast_one] using rest following anchors stack
  · refine ⟨best, env, heap, 1, by omega,
      BlockContinuesIn.range_done ?_ BlockContinuesIn.nil, ht, hm, hbest⟩
    simp [Calls.rangeContinues, before]
termination_by m - n
decreasing_by all_goals omega


theorem source_outer_loop :
    ((findFunctionIn program.body "solve").bind fun (_, body) => (executableBody body)[2]?) = some outerLoop := rfl
/-- The square potential pays for the triangular range traversal. Its decrease
is proportional to m, exactly as the source's inner range cost is. -/
theorem outerLoop_cost (total m : Nat) (best : Int) (mpositive : 2 ≤ m)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (ht : env "total" = some (.scalar (.int total))) (hm : env "m" = some (.scalar (.int m)))
    (hbest : env "best" = some (.scalar (.int best))) :
    ∃ newBest finalEnv finalHeap cost, cost ≤ 81 * (total - m * m) + 19 ∧
      BlockContinuesIn program env [outerLoop] heap cost finalEnv finalHeap ∧
      finalEnv "total" = some (.scalar (.int total)) ∧ finalEnv "best" = some (.scalar (.int newBest)) := by
  have guard : EvaluatesIn program env outerCondition heap 17
      (.scalar (.bool (decide (2 * m * (m + 1) ≤ total)))) heap := by
    have evaluated := outerCondition_cost env heap total m ht hm
    have castBound : (2 : Int) * m * ((m : Int) + 1) = (2 * m * (m + 1) : Nat) := rfl
    rw [castBound] at evaluated
    simpa only [Int.ofNat_le] using evaluated
  by_cases within : 2 * m * (m + 1) ≤ total
  · obtain ⟨rowBest, rowEnv, rowHeap, rowCost, rowBound, row, rowTotal, rowM, rowBestLookup⟩ :=
      innerRange_cost total best m 1 (by omega) env heap ht hm hbest
    have ranged : BlockContinuesIn program env
        [.forIn (.name "n") (.call (.name "range") [.constant (.int 1), .name "m"]) innerBody []]
        heap (4 + (rowCost + 1) + 1) rowEnv rowHeap :=
      BlockContinuesIn.for_range (.cons EvaluatesIn.constant
        (.cons (EvaluatesIn.lookup hm) .nil)) rfl row
    let nextEnv := HeapCalls.setLocal rowEnv "m" (.scalar (.int (m + 1 : Nat)))
    have incremented : BlockContinuesIn program rowEnv
        [.augAssign (.name "m") .add (.constant (.int 1))] rowHeap 6 nextEnv rowHeap := by
      apply BlockContinuesIn.augAssign rowM EvaluatesIn.constant
      simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
    have nextTotal : nextEnv "total" = some (.scalar (.int total)) := by
      simpa [nextEnv, HeapCalls.setLocal] using rowTotal
    have nextBest : nextEnv "best" = some (.scalar (.int rowBest)) := by
      simpa [nextEnv, HeapCalls.setLocal] using rowBestLookup
    have squareLower : m ≤ m * m := by
      have := Nat.mul_le_mul_left m (show 1 ≤ m by omega)
      simpa using this
    have nextSquare : (m + 1) * (m + 1) ≤ total := by
      simp only [Nat.mul_add, Nat.mul_one, Nat.mul_assoc] at within
      simp only [Nat.add_mul, Nat.mul_add, Nat.mul_one, Nat.one_mul]
      omega
    have potential : total - m * m = total - (m + 1) * (m + 1) + 2 * m + 1 := by
      simp only [Nat.add_mul, Nat.mul_add, Nat.mul_one, Nat.one_mul] at nextSquare ⊢
      omega
    obtain ⟨newBest, finalEnv, finalHeap, restCost, restBound, rest, keptTotal, answer⟩ :=
      outerLoop_cost total (m + 1) rowBest (by omega) nextEnv rowHeap nextTotal rfl nextBest
    refine ⟨newBest, finalEnv, finalHeap,
      17 + (((4 + (rowCost + 1) + 1) + 6) + restCost) + 2, ?_, ?_, keptTotal, answer⟩
    · rw [potential]
      omega
    · apply BlockContinuesIn.while_loop (flag := true) (value := .scalar (.bool true))
      · simpa [within] using guard
      · rfl
      · exact (ranged.append incremented).append rest
  · refine ⟨best, env, heap, 19, by omega, ?_, ht, hbest⟩
    apply BlockContinuesIn.while_loop (cc := 17) (cb := 0) (flag := false) (value := .scalar (.bool false))
    · simpa [within] using guard
    · rfl
    · exact BlockContinuesIn.nil
termination_by total + 1 - m
decreasing_by
  have bound : m ≤ total := by
    simp only [Nat.mul_add, Nat.mul_one, Nat.mul_assoc] at within
    omega
  omega

/-- The costed body is the executable body of the generated source AST. -/
theorem source_body :
    ((findFunctionIn program.body "solve").map fun (_, body) => executableBody body) =
      some ([.assign [.name "best"] (.constant (.int 0)),
        .assign [.name "m"] (.constant (.int 2)), outerLoop] ++ functionTail) := rfl

private theorem functionTail_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total best : Int)
    (ht : env "total" = some (.scalar (.int total))) (hb : env "best" = some (.scalar (.int best)))
    (anchors : Nat) :
    ∃ cost outcome, cost ≤ 18 ∧ FinishesIn (HeapCalls.step program anchors)
      ⟨.block functionTail env, [], heap⟩ cost outcome := by
  by_cases zero : best = 0
  · have test : EvaluatesIn program env (.name "best") heap 1 (.scalar (.int 0)) heap := by
      simpa only [zero] using (EvaluatesIn.lookup (module := program) (heap := heap) hb)
    have branch : BlockContinuesIn program env [.ifThen (.name "best") [.return (.name "best")] []]
        heap 3 env heap := BlockContinuesIn.branch test rfl BlockContinuesIn.nil
    have message : EvaluatesIn program env errorMessage heap 12
        (.scalar (.string (("" ++ "No Pythagorean triplet sums to ") ++ toString total))) heap := by
      apply EvaluatesIn.binary
        (EvaluatesIn.binary EvaluatesIn.constant EvaluatesIn.constant rfl)
        (EvaluatesIn.builtin (.cons (EvaluatesIn.lookup ht) .nil) rfl) rfl
    have raised : FinishesIn (HeapCalls.step program anchors) ⟨.block [raiseError] env, [], heap⟩ 15
        (.ok ⟨.error {typeName := "ValueError", args := [.string (("" ++ "No Pythagorean triplet sums to ") ++ toString total)]},
          [], heap, anchors⟩) := by
      exact .next rfl ((message anchors [.arguments (.raiseException "ValueError") [] [] env]).finishes
        (.next rfl (.done rfl)))
    exact ⟨18, _, by omega, (branch [raiseError] anchors []).finishes raised⟩
  · have returned : BlockReturnsIn program env functionTail heap 5 (.scalar (.int best)) heap := by
      apply BlockReturnsIn.branch (cc := 1) (cb := 2) (flag := true) (EvaluatesIn.lookup hb)
      · simp [Mutable.truth, scalarTruth, zero]
      · exact BlockReturnsIn.return_value (EvaluatesIn.lookup hb)
    exact ⟨6, _, by omega, (returned anchors []).finishes (.done rfl)⟩

/-- A full trace of the original `solve`, including the exception branch for
inputs without a triplet. Collection primitives and integer/string arithmetic
are atomic at the transition layer; the only sort has fixed arity three. -/
theorem timeBound (total : Nat) :
    (HeapCalls.call Solvers.P9Ast.module "solve" [.scalar (.int total)] []).execution.TimeBound
      (81 * total + 44) := by
  let initialEnv := HeapCalls.setLocal (HeapCalls.setLocal
    (HeapCalls.argumentEnv [⟨"total", some (.name "int"), none⟩] [.scalar (.int total)])
    "best" (.scalar (.int 0))) "m" (.scalar (.int 2))
  have lead : ReachesIn (HeapCalls.step program 0)
      ⟨.operation (.call "solve") [.scalar (.int total)], [], []⟩ 7
      ⟨.block ([outerLoop] ++ functionTail) initialEnv, [], []⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  obtain ⟨best, finalEnv, finalHeap, loopCost, bounded, loop, keptTotal, answer⟩ :=
    outerLoop_cost total 2 0 (by omega) initialEnv [] rfl rfl rfl
  obtain ⟨tailCost, outcome, tailBound, tail⟩ := functionTail_cost finalEnv finalHeap total best keptTotal answer 0
  refine ⟨7 + (loopCost + tailCost), outcome, lead.finishes ((loop functionTail 0 []).finishes tail), ?_⟩
  omega

/-- Linear transition complexity for the full optimized Python source on all
nonnegative totals, rather than a bound on an independent arithmetic counter. -/
theorem isBigO : Execution.IsBigO
    (fun total : Nat => (HeapCalls.call Solvers.P9Ast.module "solve" [.scalar (.int total)] []).execution)
    (fun _ => True) id id := by
  refine ⟨125, 1, ?_⟩
  intro total _ large
  apply (timeBound total).mono
  change 81 * total + 44 ≤ 125 * total
  change 1 ≤ total at large
  omega

end optimized
end solvers.«9».complexity
