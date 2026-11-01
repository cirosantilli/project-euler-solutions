import «.formalsw».solvers.«66.py.ast»
import ProjectEulerStatements.Lib.PellMinimalConvergent
import lang.py.FormalSW.Calls

namespace solvers.«66».minimal_pell_x
open FormalSW.PyAst Calls ProjectEulerStatements
open PellConvergents SqrtReducedStates
set_option Elab.async false
private abbrev program := Solvers.P66Ast.module
private abbrev number (value : Int) : Constant := .int value

def iteration : List Stmt :=
  [.assign [.name "h"] (.binOp (.binOp (.name "a") .mult (.name "h_m1")) .add (.name "h_m2")),
   .assign [.name "k"] (.binOp (.binOp (.name "a") .mult (.name "k_m1")) .add (.name "k_m2")),
   .ifThen (.compare (.binOp (.binOp (.name "h") .mult (.name "h")) .sub
     (.binOp (.binOp (.name "D") .mult (.name "k")) .mult (.name "k")))
     [(.eq, .constant (.int 1))]) [.return (.name "h")] [],
   .assign [.tuple [.name "h_m2", .name "h_m1"]] (.tuple [.name "h_m1", .name "h"]),
   .assign [.tuple [.name "k_m2", .name "k_m1"]] (.tuple [.name "k_m1", .name "k"]),
   .assign [.name "m"] (.binOp (.binOp (.name "d") .mult (.name "a")) .sub (.name "m")),
   .assign [.name "d"] (.binOp (.binOp (.name "D") .sub (.binOp (.name "m") .mult (.name "m")))
     .floorDiv (.name "d")),
   .assign [.name "a"] (.binOp (.binOp (.name "a0") .add (.name "m")) .floorDiv (.name "d"))]
def loop : Stmt := .whileDo (.constant (.bool true)) iteration []

structure Ready (n index : Nat) (env : ScalarEnv) : Prop where
  getN : env "D" = some (number n)
  getRoot : env "a0" = some (number (Nat.sqrt n))
  getM : env "m" = some (number (trajectory n index).1)
  getD : env "d" = some (number (trajectory n index).2)
  getA : env "a" = some (number (coefficient (Nat.sqrt n) (trajectory n index)))
  getOlderP : env "h_m2" = some (number (convergents n index).olderP)
  getCurrentP : env "h_m1" = some (number (convergents n index).currentP)
  getOlderQ : env "k_m2" = some (number (convergents n index).olderQ)
  getCurrentQ : env "k_m1" = some (number (convergents n index).currentQ)

theorem loop_result (n index : Nat) (nonsquare : Nat.sqrt n * Nat.sqrt n ≠ n)
    (before : index < firstHit n nonsquare) (env : ScalarEnv) (ready : Ready n index env) :
    BlockReturns program env [loop] (number (PellMinimum.minimum n)) := by
  let state := trajectory n index
  let nextState := trajectory n (index + 1)
  let values := convergents n index
  let next := convergents n (index + 1)
  let a := coefficient (Nat.sqrt n) state
  let eH := env.set "h" (number next.currentP)
  let eK := eH.set "k" (number next.currentQ)
  have evalH : Evaluates program env
      (.binOp (.binOp (.name "a") .mult (.name "h_m1")) .add (.name "h_m2"))
      (number next.currentP) := by
    exact Evaluates.binary (av := number (a * values.currentP)) (bv := number values.olderP)
      (Evaluates.binary (Evaluates.lookup ready.getA) (Evaluates.lookup ready.getCurrentP)
        (by simp [scalarBinary, number]; rfl)) (Evaluates.lookup ready.getOlderP)
      (by simp [scalarBinary, number, next, convergents, advance, a, state, values])
  have evalK : Evaluates program eH
      (.binOp (.binOp (.name "a") .mult (.name "k_m1")) .add (.name "k_m2"))
      (number next.currentQ) := by
    exact Evaluates.binary (av := number (a * values.currentQ)) (bv := number values.olderQ)
      (Evaluates.binary
        (Evaluates.lookup (by simpa [eH, ScalarEnv.set] using ready.getA))
        (Evaluates.lookup (by simpa [eH, ScalarEnv.set] using ready.getCurrentQ))
        (by simp [scalarBinary, number]; rfl))
      (Evaluates.lookup (by simpa [eH, ScalarEnv.set] using ready.getOlderQ))
      (by simp [scalarBinary, number, next, convergents, advance, a, state, values])
  have tested : Evaluates program eK
      (.compare (.binOp (.binOp (.name "h") .mult (.name "h")) .sub
        (.binOp (.binOp (.name "D") .mult (.name "k")) .mult (.name "k")))
        [(.eq, .constant (.int 1))])
      (.bool (decide (norm n next.currentP next.currentQ = 1))) := by
    apply Evaluates.compare (av := number (norm n next.currentP next.currentQ))
      (bv := number 1)
    · exact Evaluates.binary (av := number (next.currentP * next.currentP))
        (bv := number ((n : Int) * next.currentQ * next.currentQ))
        (Evaluates.binary (av := number next.currentP) (bv := number next.currentP)
          (Evaluates.lookup (by simp [eK, eH, ScalarEnv.set]))
          (Evaluates.lookup (by simp [eK, eH, ScalarEnv.set])) (by simp [scalarBinary, number]))
        (Evaluates.binary (av := number ((n : Int) * next.currentQ)) (bv := number next.currentQ)
          (Evaluates.binary (av := number n) (bv := number next.currentQ)
            (Evaluates.lookup (by simpa [eK, eH, ScalarEnv.set] using ready.getN))
            (Evaluates.lookup (by simp [eK, ScalarEnv.set])) (by simp [scalarBinary, number]))
          (Evaluates.lookup (by simp [eK, ScalarEnv.set])) (by simp [scalarBinary, number]))
        (by simp [scalarBinary, number, norm])
    · exact Evaluates.constant
    · simp [scalarCompare, evalCmpOp]
  apply BlockReturns.while_loop (value := .bool true) Evaluates.constant
  apply BlockReturns.assign evalH
  apply BlockReturns.assign evalK
  by_cases done : index + 1 = firstHit n nonsquare
  · have hit : norm n next.currentP next.currentQ = 1 := by
      simpa only [next, done] using firstHit_norm n nonsquare
    apply BlockReturns.branch (value := .bool true) (by simpa only [hit, decide_true] using tested)
    have answerEq : next.currentP = PellMinimum.minimum n := by
      simpa only [next, done] using firstHit_minimum n nonsquare
    rw [← answerEq]
    exact BlockReturns.return_value (Evaluates.lookup (by simp [eH, ScalarEnv.set]))
  · have more : index + 1 < firstHit n nonsquare := by omega
    have notHit : norm n next.currentP next.currentQ ≠ 1 :=
      before_firstHit n (index + 1) nonsquare (by omega) more
    apply BlockReturns.branch (value := .bool false) (by simpa only [notHit, decide_false] using tested)
    let eP := assignNames eK ["h_m2", "h_m1"] [number values.currentP, number next.currentP]
    let eQ := assignNames eP ["k_m2", "k_m1"] [number values.currentQ, number next.currentQ]
    let eM := eQ.set "m" (number nextState.1)
    let eD := eM.set "d" (number nextState.2)
    let eA := eD.set "a" (number (coefficient (Nat.sqrt n) nextState))
    have nextEq : step n (Nat.sqrt n) state = nextState := trajectory_step n index
    have mStep : state.2 * a - state.1 = nextState.1 := congrArg Prod.fst nextEq
    have dStep : ((n : Int) - nextState.1 ^ 2) / state.2 = nextState.2 := by
      have same := congrArg Prod.snd nextEq
      change ((n : Int) - (state.2 * a - state.1) ^ 2) / state.2 = nextState.2 at same
      simpa only [mStep] using same
    have currentPositive : 0 < state.2 := trajectory_denominator_positive n index nonsquare
    have nextPositive : 0 < nextState.2 := trajectory_denominator_positive n (index + 1) nonsquare
    apply BlockReturns.unpack (names := ["h_m2", "h_m1"])
      (results := [number values.currentP, number next.currentP]) rfl rfl
      (.cons (Evaluates.lookup (by simpa [eK, eH, ScalarEnv.set] using ready.getCurrentP))
        (.cons (Evaluates.lookup (by simp [eH, ScalarEnv.set])) .nil))
    apply BlockReturns.unpack (names := ["k_m2", "k_m1"])
      (results := [number values.currentQ, number next.currentQ]) rfl rfl
      (.cons (Evaluates.lookup (by simpa [eP, eK, eH, assignNames, ScalarEnv.set]
          using ready.getCurrentQ))
        (.cons (Evaluates.lookup rfl) .nil))
    apply BlockReturns.assign (value := number nextState.1)
    · exact Evaluates.binary (av := number (state.2 * a)) (bv := number state.1)
        (Evaluates.binary
          (Evaluates.lookup (by simpa [eQ, eP, eK, eH, assignNames, ScalarEnv.set] using ready.getD))
          (Evaluates.lookup (by simpa [eQ, eP, eK, eH, assignNames, ScalarEnv.set] using ready.getA))
          (by simp [scalarBinary, number]; rfl))
        (Evaluates.lookup (by simpa [eQ, eP, eK, eH, assignNames, ScalarEnv.set] using ready.getM))
        (by simp [scalarBinary, number, mStep])
    apply BlockReturns.assign (value := number nextState.2)
    · exact Evaluates.binary (av := number ((n : Int) - nextState.1 ^ 2)) (bv := number state.2)
        (Evaluates.binary (av := number n) (bv := number (nextState.1 ^ 2))
          (Evaluates.lookup (by simpa [eM, eQ, eP, eK, eH, assignNames, ScalarEnv.set]
            using ready.getN))
          (Evaluates.binary (av := number nextState.1) (bv := number nextState.1)
            (Evaluates.lookup (by simp [ScalarEnv.set])) (Evaluates.lookup (by simp [ScalarEnv.set]))
            (by simp [scalarBinary, number, pow_two])) (by simp [scalarBinary, number]))
        (Evaluates.lookup (by simpa [eM, eQ, eP, eK, eH, assignNames, ScalarEnv.set] using ready.getD))
        (by simp [scalarBinary, number, ne_of_gt currentPositive, currentPositive, dStep])
    apply BlockReturns.assign (value := number (coefficient (Nat.sqrt n) nextState))
    · exact Evaluates.binary (av := number ((Nat.sqrt n : Int) + nextState.1)) (bv := number nextState.2)
        (Evaluates.binary (av := number (Nat.sqrt n)) (bv := number nextState.1)
          (Evaluates.lookup (by simpa [eD, eM, eQ, eP, eK, eH, assignNames, ScalarEnv.set]
            using ready.getRoot))
          (Evaluates.lookup (by simp [ScalarEnv.set])) (by simp [scalarBinary, number]))
        (Evaluates.lookup (by simp [ScalarEnv.set]))
        (by simp [scalarBinary, number, ne_of_gt nextPositive, nextPositive, coefficient])
    have nextReady : Ready n (index + 1) eA := by
      constructor
      · simpa [eA, eD, eM, eQ, eP, eK, eH, assignNames, ScalarEnv.set] using ready.getN
      · simpa [eA, eD, eM, eQ, eP, eK, eH, assignNames, ScalarEnv.set] using ready.getRoot
      · simp [eA, eD, eM, nextState, ScalarEnv.set]
      · simp [eA, eD, nextState, ScalarEnv.set]
      · simp [eA, nextState, ScalarEnv.set]
      · simp [eA, eD, eM, eQ, eP, assignNames, ScalarEnv.set, convergents, advance, values]
      · simp [eA, eD, eM, eQ, eP, assignNames, ScalarEnv.set, next]
      · simp [eA, eD, eM, eQ, assignNames, ScalarEnv.set, convergents, advance, values]
      · simp [eA, eD, eM, eQ, assignNames, ScalarEnv.set, next]
    have rest := loop_result n (index + 1) nonsquare more eA nextReady
    simpa only [loopContinuation, show loopControl iteration = false from rfl,
      Bool.false_eq_true, if_false] using rest
termination_by firstHit n nonsquare - index
decreasing_by omega

def output (n : Nat) : Constant :=
  if Nat.sqrt n * Nat.sqrt n = n then .none else number (PellMinimum.minimum n)

theorem result (n : Nat) : ReturnsValue program "minimal_pell_x" [number n] (output n) := by
  have root_eq : Math.isqrt n = Nat.sqrt n := Nat.eq_sqrt.mpr (Math.isqrt_spec n)
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := number (Nat.sqrt n))
  · apply Evaluates.qualified_call
    apply Evaluates.library_call (.cons (Evaluates.lookup rfl) .nil) rfl
    change some (Math.isqrtCall [.int n]) = _
    simp [Math.isqrtCall, root_eq, show ¬ (n : Int) < 0 by omega]
  let env := (argumentEnv [{ name := "D", annotation := some (.name "int") }]
    [number n]).set "a0" (number (Nat.sqrt n))
  have squareTest : Evaluates program env
      (.compare (.binOp (.name "a0") .mult (.name "a0")) [(.eq, .name "D")])
      (.bool (decide (Nat.sqrt n * Nat.sqrt n = n))) := by
    apply Evaluates.compare (av := number ((Nat.sqrt n * Nat.sqrt n : Nat) : Int)) (bv := number n)
      (Evaluates.binary (av := number (Nat.sqrt n)) (bv := number (Nat.sqrt n))
        (Evaluates.lookup rfl) (Evaluates.lookup rfl) (by simp [scalarBinary, number]))
      (Evaluates.lookup rfl)
    simp [scalarCompare, evalCmpOp]
    norm_cast
  by_cases square : Nat.sqrt n * Nat.sqrt n = n
  · apply BlockReturns.branch (value := .bool true) (by simpa only [square, decide_true] using squareTest)
    simp only [output, if_pos square]
    exact BlockReturns.return_value Evaluates.constant
  · apply BlockReturns.branch (value := .bool false) (by simpa only [square, decide_false] using squareTest)
    simp only [output, if_neg square]
    apply BlockReturns.assign (value := number 0) Evaluates.constant
    apply BlockReturns.assign (value := number 1) Evaluates.constant
    apply BlockReturns.assign (value := number (Nat.sqrt n)) (Evaluates.lookup rfl)
    apply BlockReturns.unpack (names := ["h_m2", "h_m1"]) (results := [number 0, number 1])
      rfl rfl (.cons Evaluates.constant (.cons Evaluates.constant .nil))
    apply BlockReturns.unpack (names := ["k_m2", "k_m1"]) (results := [number 1, number 0])
      rfl rfl (.cons Evaluates.constant (.cons Evaluates.constant .nil))
    apply loop_result n 0 square (firstHit_positive n square)
    constructor
    · rfl
    all_goals simp [ScalarEnv.set, assignNames, convergents, initial, trajectory, coefficient]

end solvers.«66».minimal_pell_x
