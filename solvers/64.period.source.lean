import «.formalsw».solvers.«64.py.ast»
import ProjectEulerStatements.P64
import lang.py.FormalSW.Calls

namespace solvers.«64».period_length_sqrt
open FormalSW.PyAst Calls ProjectEulerStatements.SqrtReducedStates
set_option Elab.async false
private abbrev program := Solvers.P64Ast.module
private abbrev number (value : Int) : Constant := .int value

def iteration : List Stmt :=
  [.assign [.name "m"] (.binOp (.binOp (.name "d") .mult (.name "a")) .sub (.name "m")),
   .assign [.name "d"] (.binOp (.binOp (.name "n") .sub (.binOp (.name "m") .mult (.name "m")))
     .floorDiv (.name "d")),
   .assign [.name "a"] (.binOp (.binOp (.name "a0") .add (.name "m")) .floorDiv (.name "d")),
   .augAssign (.name "period") .add (.constant (.int 1)),
   .ifThen (.compare (.name "a") [(.eq, .binOp (.constant (.int 2)) .mult (.name "a0"))])
     [.return (.name "period")] []]
def loop : Stmt := .whileDo (.constant (.bool true)) iteration []

/-- The actual unbounded source loop stops exactly at the first terminal reduced state. -/
theorem loop_result (n index : Nat) (nonsquare : Nat.sqrt n * Nat.sqrt n ≠ n)
    (before : index < length n) (env : ScalarEnv)
    (getN : env "n" = some (number n)) (getRoot : env "a0" = some (number (Nat.sqrt n)))
    (getM : env "m" = some (number (trajectory n index).1))
    (getD : env "d" = some (number (trajectory n index).2))
    (getA : env "a" = some (number (coefficient (Nat.sqrt n) (trajectory n index))))
    (getCount : env "period" = some (number index)) (following : List Stmt) :
    BlockReturns program env (loop :: following) (number (length n)) := by
  let state := trajectory n index
  let next := trajectory n (index + 1)
  have nextEq : step n (Nat.sqrt n) state = next := trajectory_step n index
  have mStep : state.2 * coefficient (Nat.sqrt n) state - state.1 = next.1 := congrArg Prod.fst nextEq
  have dStep : ((n : Int) - next.1 ^ 2) / state.2 = next.2 := by
    have same := congrArg Prod.snd nextEq
    change ((n : Int) - (state.2 * coefficient (Nat.sqrt n) state - state.1) ^ 2) / state.2 = next.2 at same
    simpa only [mStep] using same
  have currentNonzero : state.2 ≠ 0 := ne_of_gt (trajectory_denominator_positive n index nonsquare)
  have nextNonzero : next.2 ≠ 0 := ne_of_gt (trajectory_denominator_positive n (index + 1) nonsquare)
  have currentPositive : 0 < state.2 := trajectory_denominator_positive n index nonsquare
  have nextPositive : 0 < next.2 := trajectory_denominator_positive n (index + 1) nonsquare
  let e1 := env.set "m" (number next.1)
  let e2 := e1.set "d" (number next.2)
  let e3 := e2.set "a" (number (coefficient (Nat.sqrt n) next))
  let e4 := e3.set "period" (number ((index + 1 : Nat) : Int))
  have tested : Evaluates program e4
      (.compare (.name "a") [(.eq, .binOp (.constant (.int 2)) .mult (.name "a0"))])
      (.bool (decide (coefficient (Nat.sqrt n) next = 2 * (Nat.sqrt n : Int)))) := by
    apply Evaluates.compare (av := number (coefficient (Nat.sqrt n) next)) (bv := number (2 * (Nat.sqrt n : Int)))
      (Evaluates.lookup (by simp [e4, e3, ScalarEnv.set]))
    · exact Evaluates.binary Evaluates.constant
        (Evaluates.lookup (by simpa [e4, e3, e2, e1, ScalarEnv.set] using getRoot))
        (by simp [scalarBinary, number])
    · simp [scalarCompare, evalCmpOp]
  apply BlockReturns.while_loop (value := .bool true) Evaluates.constant
  simp only [scalarTruth, if_true]
  apply BlockReturns.assign (value := number next.1)
  · exact Evaluates.binary (av := number (state.2 * coefficient (Nat.sqrt n) state)) (bv := number state.1)
      (Evaluates.binary (av := number state.2) (bv := number (coefficient (Nat.sqrt n) state))
        (Evaluates.lookup getD) (Evaluates.lookup getA) (by simp [scalarBinary, number]))
      (Evaluates.lookup getM) (by simp [scalarBinary, number, mStep])
  apply BlockReturns.assign (value := number next.2)
  · exact Evaluates.binary (av := number ((n : Int) - next.1 ^ 2)) (bv := number state.2)
      (Evaluates.binary (av := number n) (bv := number (next.1 ^ 2))
        (Evaluates.lookup (by simpa [e1, ScalarEnv.set] using getN))
        (Evaluates.binary (av := number next.1) (bv := number next.1)
          (Evaluates.lookup (by simp [ScalarEnv.set])) (Evaluates.lookup (by simp [ScalarEnv.set]))
          (by simp [scalarBinary, number, pow_two])) (by simp [scalarBinary, number]))
      (Evaluates.lookup (by simpa [e1, ScalarEnv.set] using getD))
      (by simp [scalarBinary, number, currentNonzero, currentPositive, dStep])
  apply BlockReturns.assign (value := number (coefficient (Nat.sqrt n) next))
  · exact Evaluates.binary (av := number ((Nat.sqrt n : Int) + next.1)) (bv := number next.2)
      (Evaluates.binary (av := number (Nat.sqrt n)) (bv := number next.1)
        (Evaluates.lookup (by simpa [e2, e1, ScalarEnv.set] using getRoot))
        (Evaluates.lookup (by simp [ScalarEnv.set])) (by simp [scalarBinary, number]))
      (Evaluates.lookup (by simp [ScalarEnv.set]))
      (by simp [scalarBinary, number, nextNonzero, nextPositive, coefficient])
  apply BlockReturns.augAssign (value := number ((index + 1 : Nat) : Int))
    (Evaluates.binary (Evaluates.lookup (by simpa [e3, e2, e1, ScalarEnv.set] using getCount))
      Evaluates.constant (by simp [scalarBinary, number]))
  by_cases done : index + 1 = length n
  · have endpoint : coefficient (Nat.sqrt n) next = 2 * (Nat.sqrt n : Int) := by
      dsimp only [next]
      rw [trajectory_succ, show index = length n - 1 by omega]
      exact source_stops n nonsquare
    have decision : decide (coefficient (Nat.sqrt n) next = 2 * (Nat.sqrt n : Int)) = true :=
      decide_eq_true endpoint
    rw [decision] at tested
    apply BlockReturns.branch (value := .bool true) tested
    apply BlockReturns.return_value
    simpa only [done] using (Evaluates.lookup (module := program)
      (show e4 "period" = some (number ((index + 1 : Nat) : Int)) by simp [e4, ScalarEnv.set]))
  · have more : index + 1 < length n := by omega
    have different : coefficient (Nat.sqrt n) next ≠ 2 * (Nat.sqrt n : Int) := by
      simpa only [next, trajectory_succ] using source_continues n index nonsquare more
    have decision : decide (coefficient (Nat.sqrt n) next = 2 * (Nat.sqrt n : Int)) = false :=
      decide_eq_false different
    rw [decision] at tested
    apply BlockReturns.branch (value := .bool false) tested
    have rest := loop_result n (index + 1) nonsquare more e4
      (by simpa [e4, e3, e2, e1, ScalarEnv.set] using getN)
      (by simpa [e4, e3, e2, e1, ScalarEnv.set] using getRoot)
      (by simp [next, e4, e3, e2, e1, ScalarEnv.set]) (by simp [next, e4, e3, e2, ScalarEnv.set])
      (by simp [next, e4, e3, ScalarEnv.set]) (by simp [e4, ScalarEnv.set]) following
    simpa only [Calls.loopContinuation, show Calls.loopControl iteration = false from rfl,
      Bool.false_eq_true, if_false] using rest
termination_by length n - index
decreasing_by omega

theorem result (n : Nat) :
    ReturnsValue program "period_length_sqrt" [number n] (number (ProjectEulerStatements.P64.periodLen n)) := by
  have root_eq : Math.isqrt n = Nat.sqrt n := Nat.eq_sqrt.mpr (Math.isqrt_spec n)
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := number (Nat.sqrt n))
  · apply Evaluates.library_call (.cons (Evaluates.lookup rfl) .nil) rfl
    change some (Math.isqrtCall [.int n]) = _
    simp [Math.isqrtCall, root_eq, show ¬ (n : Int) < 0 by omega]
  by_cases square : Nat.sqrt n * Nat.sqrt n = n
  · apply BlockReturns.branch (value := .bool true)
    · exact Evaluates.compare (av := number ((Nat.sqrt n * Nat.sqrt n : Nat) : Int)) (bv := number n)
        (Evaluates.binary (av := number (Nat.sqrt n)) (bv := number (Nat.sqrt n))
          (Evaluates.lookup rfl) (Evaluates.lookup rfl) (by simp [scalarBinary, number]))
        (Evaluates.lookup rfl) (by simp [scalarCompare, evalCmpOp]; exact_mod_cast square)
    rw [ProjectEulerStatements.P64.periodLen_of_square n square]
    exact BlockReturns.return_value Evaluates.constant
  · apply BlockReturns.branch (value := .bool false)
    · exact Evaluates.compare (av := number ((Nat.sqrt n * Nat.sqrt n : Nat) : Int)) (bv := number n)
        (Evaluates.binary (av := number (Nat.sqrt n)) (bv := number (Nat.sqrt n))
          (Evaluates.lookup rfl) (Evaluates.lookup rfl) (by simp [scalarBinary, number]))
        (Evaluates.lookup rfl) (by simp [scalarCompare, evalCmpOp]; exact_mod_cast square)
    apply BlockReturns.assign (value := number 0) Evaluates.constant
    apply BlockReturns.assign (value := number 1) Evaluates.constant
    apply BlockReturns.assign (value := number (Nat.sqrt n)) (Evaluates.lookup rfl)
    apply BlockReturns.assign (value := number 0) Evaluates.constant
    exact loop_result n 0 square (length_positive n square) _ rfl rfl rfl rfl
      (by simp [trajectory, coefficient, ScalarEnv.set]) rfl []

end solvers.«64».period_length_sqrt
