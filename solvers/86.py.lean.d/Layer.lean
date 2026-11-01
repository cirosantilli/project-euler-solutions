import solvers.«86.py.lean.d».Arithmetic
import ProjectEulerSolutions.Lib.NaturalIntervalSum

namespace solvers.«86»
open FormalSW.PyAst Calls ProjectEulerSolutions.CuboidRoute
open scoped BigOperators
set_option Elab.async false

def contribution (c s : Nat) : Nat := if square (s ^ 2 + c ^ 2) then pairCount c s else 0
def remaining (c first : Nat) : Nat := ∑ s ∈ Finset.Icc first (2 * c), contribution c s

abbrev layerBody : List Stmt :=
  [.assign [.name "d"] distanceExpression,
   .assign [.name "t"] (.call (.name "isqrt") [.name "d"]),
   .ifThen squareCondition
     [.assign [.name "lo"] lowerExpression, .assign [.name "hi"] upperExpression,
      .augAssign (.name "total") .add additionExpression] []]

structure Locals (env : ScalarEnv) (c acc : Nat) : Prop where
  side : env "c" = some (number c)
  squared : env "c2" = some (number (c ^ 2))
  count : env "total" = some (number acc)

theorem layer_result (c s acc : Nat) (lower : 2 ≤ s) (env : ScalarEnv) (ready : Locals env c acc)
    (following : List Stmt) (answer : Constant)
    (rest : ∀ finalEnv, Locals finalEnv c (acc + remaining c s) → BlockReturns program finalEnv following answer) :
    BlockReturns program env
      (.rangeStepLoop "s" s (2 * c + 1) 1 layerBody [] :: following) answer := by
  by_cases inside : s ≤ 2 * c
  · let entered := env.set "s" (number s)
    let measured := entered.set "d" (number (s ^ 2 + c ^ 2))
    let rooted := measured.set "t" (number (Nat.sqrt (s ^ 2 + c ^ 2)))
    have getS : entered "s" = some (number s) := by simp [entered, ScalarEnv.set]
    have getC2 : entered "c2" = some (number (c ^ 2)) := by simpa [entered, ScalarEnv.set] using ready.squared
    have rootedReady : Locals rooted c acc := by
      rcases ready with ⟨hc, hc2, ha⟩
      constructor <;> simp_all [rooted, measured, entered, ScalarEnv.set]
    have rootedS : rooted "s" = some (number s) := by simp [rooted, measured, entered, ScalarEnv.set]
    have tested := square_evaluate (s ^ 2 + c ^ 2) rooted
      (by simp [rooted, measured, ScalarEnv.set]) (by simp [rooted, ScalarEnv.set])
    have sumStep := ProjectEulerSolutions.NaturalIntervalSum.step (contribution c) s (2 * c) inside
    change remaining c s = contribution c s + remaining c (s + 1) at sumStep
    apply BlockReturns.range_step (by simp [rangeContinues]; omega)
    apply BlockReturns.assign (distance_evaluate c s entered getS getC2)
    apply BlockReturns.assign (root_evaluate (s ^ 2 + c ^ 2) measured (by simp [measured, ScalarEnv.set]))
    by_cases accepted : square (s ^ 2 + c ^ 2)
    · let low := rooted.set "lo" (.int (max 1 ((s : Int) - c)))
      let high := low.set "hi" (number (s / 2))
      let updated := high.set "total" (number (acc + pairCount c s))
      have nextReady : Locals updated c (acc + pairCount c s) := by
        rcases rootedReady with ⟨hc, hc2, ha⟩
        constructor <;> simp_all [updated, high, low, ScalarEnv.set]
      have continued := layer_result c (s + 1) (acc + pairCount c s) (by omega) updated nextReady
        following answer (fun finalEnv finalReady => rest finalEnv (by
          simpa only [sumStep, contribution, if_pos accepted, Nat.add_assoc] using finalReady))
      apply BlockReturns.branch (value := .bool true) (by simpa only [accepted, decide_true] using tested)
      apply BlockReturns.assign (lower_evaluate c s rooted rootedS rootedReady.side)
      apply BlockReturns.assign (upper_evaluate s low (by simpa [low, ScalarEnv.set] using rootedS))
      apply BlockReturns.augAssign (value := number (acc + pairCount c s))
      · exact Evaluates.binary (av := number acc) (bv := number (pairCount c s))
          (Evaluates.lookup (by simpa [high, low, ScalarEnv.set] using rootedReady.count))
          (addition_evaluate c s lower inside high (by simp [high, low, ScalarEnv.set])
            (by simp [high, ScalarEnv.set])) (by simp [scalarBinary, number])
      exact continued
    · have continued := layer_result c (s + 1) acc (by omega) rooted rootedReady following answer
        (fun finalEnv finalReady => rest finalEnv (by
          simpa only [sumStep, contribution, if_neg accepted, Nat.zero_add] using finalReady))
      apply BlockReturns.branch (value := .bool false) (by simpa only [accepted, decide_false] using tested)
      exact continued
  · have empty := ProjectEulerSolutions.NaturalIntervalSum.empty (contribution c) s (2 * c) (by omega)
    apply BlockReturns.range_done (by simp [rangeContinues]; omega)
    apply rest env
    simpa only [remaining, empty, Nat.add_zero] using ready
termination_by 2 * c + 1 - s
decreasing_by all_goals omega

/-- The actual sum loop counts each eligible ordered side pair once. -/
theorem count_new_solutions.result (c : Nat) :
    ReturnsValue program "count_new_solutions" [number c] (number (groupedLayer c)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := number 0) Evaluates.constant
  apply BlockReturns.assign (value := number (c ^ 2))
    (Evaluates.binary (av := number c) (bv := number c) (Evaluates.lookup rfl) (Evaluates.lookup rfl)
      (by simp [scalarBinary, number, pow_two]))
  apply BlockReturns.for_range (values := [number 2, number (2 * c + 1)])
  · exact .cons Evaluates.constant (.cons
      (Evaluates.binary (av := number (2 * c)) (bv := number 1)
        (Evaluates.binary (av := number 2) (bv := number c) Evaluates.constant (Evaluates.lookup rfl)
          (by simp [scalarBinary, number])) Evaluates.constant (by simp [scalarBinary, number])) .nil)
  · rfl
  apply layer_result c 2 0 (by omega) _ ⟨rfl, rfl, rfl⟩
  intro finalEnv finalReady
  apply BlockReturns.return_value
  exact Evaluates.lookup (by simpa only [Nat.zero_add, remaining, contribution, groupedLayer] using finalReady.count)

end solvers.«86»
