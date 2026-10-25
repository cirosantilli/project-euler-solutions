import solvers.«66.minimum.source»
import ProjectEulerStatements.Lib.PellSearch

namespace solvers.«66».solve
open FormalSW.PyAst Calls ProjectEulerStatements.PellMinimum ProjectEulerStatements.PellSearch
set_option Elab.async false
private abbrev program := Solvers.P66Ast.module
private abbrev natural (n : Nat) : Constant := .int n
private abbrev number (n : Int) : Constant := .int n

def iteration : List Stmt :=
  [.assign [.name "x"] (.call (.name "minimal_pell_x") [.name "D"]),
   .ifThen (.builtinCall .identity [.name "x", .constant .none]) [.continueLoop] [],
   .ifThen (.compare (.name "x") [(.gt, .name "best_x")])
     [.assign [.name "best_x"] (.name "x"), .assign [.name "best_d"] (.name "D")] []]

theorem loop_result (stop : Int) (first size bestD bestX : Nat) (window : (stop - first).toNat = size)
    (env : ScalarEnv)
    (getBestD : env "best_d" = some (natural bestD)) (getBestX : env "best_x" = some (natural bestX)) :
    BlockReturns program env
      [.rangeStepLoop "D" first stop 1 iteration [], .return (.name "best_d")]
      (natural (scan first size bestD bestX).1) := by
  induction size generalizing first bestD bestX env with
  | zero =>
      apply BlockReturns.range_done (by simp [rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup getBestD)
  | succ size ih =>
      let entered := env.set "D" (natural first)
      let ready := entered.set "x" (minimal_pell_x.output first)
      have call : Evaluates program entered (.call (.name "minimal_pell_x") [.name "D"])
          (minimal_pell_x.output first) :=
        Evaluates.call (.cons (Evaluates.lookup (by simp [entered, ScalarEnv.set])) .nil)
          (minimal_pell_x.result first)
      by_cases square : Nat.sqrt first * Nat.sqrt first = first
      · rw [scan_succ, minimum_of_square first square, if_neg (by omega : ¬ bestX < 0)]
        have rest := ih (first + 1) bestD bestX (by omega) ready
          (by simpa [ready, entered, ScalarEnv.set] using getBestD)
          (by simpa [ready, entered, ScalarEnv.set] using getBestX)
        apply BlockReturns.range_step (by simp [rangeContinues]; omega)
        apply BlockReturns.assign call
        apply BlockReturns.branch (value := .bool true)
        · exact Evaluates.builtin (values := [.none, .none])
            (.cons (Evaluates.lookup (by simp [minimal_pell_x.output, square, ScalarEnv.set]))
              (.cons Evaluates.constant .nil)) rfl
        · apply BlockReturns.continue_loop rfl
          simpa only [Int.natCast_add, Int.natCast_one] using rest
      · have outputEq : minimal_pell_x.output first = natural (minimum first) := by
          simp [minimal_pell_x.output, square, natural]
        have tested : Evaluates program ready (.compare (.name "x") [(.gt, .name "best_x")])
            (.bool (decide (bestX < minimum first))) := by
          apply Evaluates.compare (av := natural (minimum first)) (bv := natural bestX)
            (Evaluates.lookup (by simp [ready, outputEq, ScalarEnv.set]))
            (Evaluates.lookup (by simpa [ready, entered, ScalarEnv.set] using getBestX))
          simp [scalarCompare, evalCmpOp]
        apply BlockReturns.range_step (by simp [rangeContinues]; omega)
        apply BlockReturns.assign call
        apply BlockReturns.branch (value := .bool false)
        · exact Evaluates.builtin (values := [natural (minimum first), .none])
            (.cons (Evaluates.lookup (by simp [outputEq, ScalarEnv.set]))
              (.cons Evaluates.constant .nil)) rfl
        · by_cases better : bestX < minimum first
          · rw [scan_succ, if_pos better]
            let improved := (ready.set "best_x" (natural (minimum first))).set "best_d" (natural first)
            have rest := ih (first + 1) first (minimum first) (by omega) improved
              (by simp [improved, ScalarEnv.set]) (by simp [improved, ScalarEnv.set])
            apply BlockReturns.branch (value := .bool true)
              (by simpa only [better, decide_true] using tested)
            apply BlockReturns.assign (value := natural (minimum first))
              (Evaluates.lookup (by simp [outputEq, ScalarEnv.set]))
            apply BlockReturns.assign (value := natural first)
              (Evaluates.lookup (by simp [entered, ScalarEnv.set]))
            apply BlockReturns.loop_resume
            simpa only [Int.natCast_add, Int.natCast_one] using rest
          · rw [scan_succ, if_neg better]
            have rest := ih (first + 1) bestD bestX (by omega) ready
              (by simpa [ready, entered, ScalarEnv.set] using getBestD)
              (by simpa [ready, entered, ScalarEnv.set] using getBestX)
            apply BlockReturns.branch (value := .bool false)
              (by simpa only [better, decide_false] using tested)
            apply BlockReturns.loop_resume
            simpa only [Int.natCast_add, Int.natCast_one] using rest

theorem body_result (limit : Int) (env : ScalarEnv) (getLimit : env "limit" = some (number limit)) :
    BlockReturns program env
      [.assign [.name "best_d"] (.constant (.int 0)),
       .assign [.name "best_x"] (.constant (.int 0)),
       .forIn (.name "D") (.call (.name "range")
         [.constant (.int 2), .binOp (.name "limit") .add (.constant (.int 1))]) iteration [],
       .return (.name "best_d")]
      (natural (answer limit)) := by
  apply BlockReturns.assign (value := natural 0) Evaluates.constant
  apply BlockReturns.assign (value := natural 0) Evaluates.constant
  apply BlockReturns.for_range (values := [natural 2, number (limit + 1)])
    (.cons Evaluates.constant (.cons (Evaluates.binary (av := number limit) (bv := number 1)
      (Evaluates.lookup (by simpa [ScalarEnv.set] using getLimit)) Evaluates.constant
      (by simp [scalarBinary, number])) .nil)) rfl
  exact loop_result (limit + 1) 2 (limit - 1).toNat 0 0 (by omega) _ rfl rfl

theorem result (limit : Int) : ReturnsValue program "solve" [number limit] (natural (answer limit)) := by
  apply BlockReturns.function rfl rfl
  exact body_result limit _ rfl

theorem default_result : ReturnsValue program "solve" [] (natural (answer 1000)) := by
  apply BlockReturns.function_defaults rfl (by decide) rfl
  exact body_result 1000 _ rfl

end solvers.«66».solve
