import solvers.«75.py.lean.d».Sieve
import ProjectEulerStatements.P75

/-! The live filtered sum and the complete signed optimized Python call. -/
namespace solvers.«75».solve
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

abbrev chosenTest : Expr := .compare (.name "c") [(.eq, .constant (.int 1))]
abbrev sumLoopBody : List Stmt :=
  comprehensionFilters [chosenTest] (sumBody (.constant (.int 1)) [])

def ones (values : List Nat) : Nat := (values.filter (fun value => decide (value = 1))).length

theorem ones_fold (values : List Nat) (initial : Nat) :
    values.foldl (fun total value => if value = 1 then total+1 else total) initial =
      initial + ones values := by
  induction values generalizing initial with
  | nil => rfl
  | cons value rest ih =>
      by_cases chosen : value = 1 <;> simp [ones, chosen, ih, Nat.add_assoc, Nat.add_comm]

theorem ones_table (limit : Nat) :
    ones ((List.range (limit+1)).map
      (ProjectEulerStatements.PythagoreanPerimeterCounting.table limit)) =
      ProjectEulerStatements.P75.countSingular limit := by
  have folded := (ones_fold ((List.range (limit+1)).map
    (ProjectEulerStatements.PythagoreanPerimeterCounting.table limit)) 0).symm
  simpa only [Nat.zero_add, List.foldl_map,
    ProjectEulerStatements.P75.countSingular_table] using folded

theorem sum_scan (values remaining : List Nat) (address position total : Nat)
    (suffix : values.drop position = remaining) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (values.map number)))
    (getTotal : env sumAccumulator = some (number total)) :
    BlockReturns program env
      [.sequenceStepLoop "c" (.ref address) position sumLoopBody [],
        .return (.name sumAccumulator)] heap (number (total + ones remaining)) heap := by
  induction remaining generalizing position total env with
  | nil =>
      have missing : values[position]? = none := by rw [← List.head?_drop, suffix]; rfl
      apply BlockReturns.sequence_done (items := values.map number)
        (by simp [sequenceValue, Mutable.sequence, found])
        (by simp [List.getElem?_map, missing])
      exact BlockReturns.return_value (by simpa [ones] using Evaluates.lookup getTotal)
  | cons value rest ih =>
      have selected : values[position]? = some value := by rw [← List.head?_drop, suffix]; rfl
      have nextSuffix : values.drop (position+1) = rest := by
        simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons]
      let entered := setLocal env "c" (number value)
      have enteredTotal : entered sumAccumulator = some (number total) := by
        simpa [entered, setLocal, sumAccumulator] using getTotal
      have tested : Evaluates program entered chosenTest heap
          (.scalar (.bool (decide (value = 1)))) heap :=
        Evaluates.compare (av := number value) (bv := number 1)
          (Evaluates.lookup (by simp [entered, setLocal])) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
      apply BlockReturns.sequence_step (items := values.map number) (value := number value)
        (by simp [sequenceValue, Mutable.sequence, found])
        (by simp [List.getElem?_map, selected])
      by_cases chosen : value = 1
      · let updated := setLocal entered sumAccumulator (number (total+1))
        have walked := ih (position+1) (total+1) nextSuffix updated
          (by simp [updated, setLocal])
        apply BlockReturns.branch (flag := true)
          (by simpa only [entered, chosen, decide_true] using tested) rfl
        apply BlockReturns.assign (value := number (total+1)) (middle := heap)
        · exact Evaluates.binary (Evaluates.lookup enteredTotal) Evaluates.constant
            (by simp [Mutable.binary, scalarBinary, number, integer])
        · simpa [ones, chosen, sumLoopBody, comprehensionFilters, sumBody,
            Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            entered, updated, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using walked
      · have walked := ih (position+1) total nextSuffix entered enteredTotal
        apply BlockReturns.branch (flag := false)
          (by simpa only [entered, chosen, decide_false] using tested) rfl
        simpa [ones, chosen, sumLoopBody, comprehensionFilters, sumBody,
          Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, entered] using walked

theorem final_sum (limit : Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getCounts : env "counts" = some (.ref base.length)) :
    BlockReturns program env (code.drop 4)
      (arrayHeap base limit (ProjectEulerStatements.PythagoreanPerimeterCounting.table limit))
      (number (ProjectEulerStatements.P75.countSingular limit))
      (arrayHeap base limit (ProjectEulerStatements.PythagoreanPerimeterCounting.table limit)) := by
  let values := (List.range (limit+1)).map
    (ProjectEulerStatements.PythagoreanPerimeterCounting.table limit)
  let generators : List (String × Expr × List Expr) := [("c", .name "counts", [chosenTest])]
  let ready := setLocal (comprehensionEnv env generators) sumAccumulator (number 0)
  have found : (arrayHeap base limit
      (ProjectEulerStatements.PythagoreanPerimeterCounting.table limit))[base.length]? =
      some (.list (values.map number)) := by simp [arrayHeap, table, values, List.map_map]
  have counted := sum_scan values values base.length 0 0 (by rfl)
    (arrayHeap base limit (ProjectEulerStatements.PythagoreanPerimeterCounting.table limit))
    ready found (by rfl)
  simp only [Nat.zero_add, values, ones_table] at counted
  apply BlockReturns.return_value
  apply Evaluates.sum_generator_sequence (items := values.map number)
    (by intros; intro h; cases h) (Evaluates.lookup getCounts)
    (by simp [Mutable.sequence, found]) Evaluates.constant rfl
  exact counted

/-- All signed bounds execute the source; negative bounds allocate no counter table. -/
theorem result (limit : Int) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [integer limit] heap
      (integer (ProjectEulerStatements.P75.result limit)) finalHeap := by
  let definition := (findFunctionIn program.body "solve").getD ([], [])
  let initial := functionEnv definition.1 definition.2 [integer limit] []
  have initialLimit : initial "limit" = some (integer limit) := rfl
  have tested : Evaluates program initial
      (.compare (.name "limit") [(.lt, .constant (.int 0))]) heap
      (.scalar (.bool (decide (limit < 0)))) heap :=
    Evaluates.compare (Evaluates.lookup initialLimit) Evaluates.constant rfl
  by_cases negative : limit < 0
  · refine ⟨heap, ?_⟩
    rw [ProjectEulerStatements.P75.result_negative limit negative]
    apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
    apply BlockReturns.branch (flag := true)
      (by simpa only [negative, decide_true] using tested) rfl
    exact BlockReturns.return_value Evaluates.constant
  · lift limit to Nat using (by omega : 0 ≤ limit)
    obtain ⟨finalEnv, executed, getCounts⟩ := sieve limit heap initial initialLimit
    let base := heap ++ [.list [number 0]]
    let finalHeap := arrayHeap base limit
      (ProjectEulerStatements.PythagoreanPerimeterCounting.table limit)
    have counted := final_sum limit base finalEnv
      (by simpa only [base, List.length_append, List.length_singleton] using getCounts)
    refine ⟨finalHeap, ?_⟩
    rw [ProjectEulerStatements.P75.result_nat]
    apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
    apply BlockReturns.branch (flag := false)
      (by simpa only [negative, decide_false] using tested) rfl
    have full := executed.returns counted
    exact full

end solvers.«75».solve
