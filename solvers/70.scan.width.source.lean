import solvers.«70.scan.conditions.source»
import ProjectEulerStatements.Lib.TotientPermutationScan

/-! The source boundary assignments maintain the moving decimal powers. -/
namespace solvers.«70».scan
open FormalSW.PyAst HeapCalls ProjectEulerStatements TotientPermutation
set_option Elab.async false

structure Ready (limit n best bestPhi width address : Nat) (env : HeapCalls.Env) : Prop where
  getLimit : env "limit" = some (number limit)
  getN : env "n" = some (number n)
  getBest : env "best_n" = some (number best)
  getBestPhi : env "best_phi" = some (number bestPhi)
  getLow : env "low" = some (number (10 ^ width : Nat))
  getHigh : env "high" = some (number (10 ^ (width + 1) : Nat))
  getPhi : env "phi" = some (.ref address)

abbrev boundaryCondition : Expr := .compare (.name "n") [(.eq, .name "high")]
abbrev boundaryCode : Stmt := .ifThen boundaryCondition
  [.assign [.name "low"] (.name "high"), .augAssign (.name "high") .mult (.constant (.int 10))] []

theorem width_code_source : scanBody[0]? = some boundaryCode := rfl

theorem Ready.at_n {limit n best bestPhi width address : Nat} {env : HeapCalls.Env}
    (ready : Ready limit n best bestPhi width address env) (next : Nat) :
    Ready limit next best bestPhi width address (setLocal env "n" (number next)) where
  getLimit := by simpa [setLocal] using ready.getLimit
  getN := by simp [setLocal]
  getBest := by simpa [setLocal] using ready.getBest
  getBestPhi := by simpa [setLocal] using ready.getBestPhi
  getLow := by simpa [setLocal] using ready.getLow
  getHigh := by simpa [setLocal] using ready.getHigh
  getPhi := by simpa [setLocal] using ready.getPhi

theorem Ready.set_pn {limit n best bestPhi width address : Nat} {env : HeapCalls.Env}
    (ready : Ready limit n best bestPhi width address env) (pn : Nat) :
    Ready limit n best bestPhi width address (setLocal env "pn" (number pn)) where
  getLimit := by simpa [setLocal] using ready.getLimit
  getN := by simpa [setLocal] using ready.getN
  getBest := by simpa [setLocal] using ready.getBest
  getBestPhi := by simpa [setLocal] using ready.getBestPhi
  getLow := by simpa [setLocal] using ready.getLow
  getHigh := by simpa [setLocal] using ready.getHigh
  getPhi := by simpa [setLocal] using ready.getPhi

theorem boundary_condition_result (limit n best bestPhi width address : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (ready : Ready limit n best bestPhi width address env) :
    Evaluates program env boundaryCondition heap
      (flag (decide (n = 10 ^ (width + 1)))) heap := by
  exact Evaluates.compare (middle := heap) (afterRight := heap)
    (Evaluates.lookup ready.getN) (Evaluates.lookup ready.getHigh)
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; norm_cast)

/-- Crossing an upper endpoint changes both powers and preserves every other scan local. -/
theorem width_result (limit n best bestPhi width address : Nat) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (ready : Ready limit n best bestPhi width address env) :
    ∃ finalEnv, BlockContinues program env [boundaryCode] heap finalEnv heap ∧
      Ready limit n best bestPhi (advanceWidth n width) address finalEnv := by
  have tested := boundary_condition_result limit n best bestPhi width address heap env ready
  by_cases boundary : n = 10 ^ (width + 1)
  · let lowerEnv := setLocal env "low" (number (10 ^ (width + 1) : Nat))
    let raisedEnv := setLocal lowerEnv "high" (number (10 ^ (width + 1) * 10 : Nat))
    have assigned : BlockContinues program env [.assign [.name "low"] (.name "high")]
        heap lowerEnv heap := BlockContinues.assign (Evaluates.lookup ready.getHigh)
    have multiplied : BlockContinues program lowerEnv
        [.augAssign (.name "high") .mult (.constant (.int 10))] heap raisedEnv heap :=
      BlockContinues.augAssign (by simpa [lowerEnv, setLocal] using ready.getHigh)
        Evaluates.constant (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary])
    refine ⟨raisedEnv, ?_, ?_⟩
    · apply BlockContinues.branch (flag := true)
        (by simpa only [boundary, decide_true] using tested) rfl
      exact assigned.append multiplied
    · simp only [advanceWidth, if_pos boundary]
      constructor
      · simpa [raisedEnv, lowerEnv, setLocal] using ready.getLimit
      · simpa [raisedEnv, lowerEnv, setLocal] using ready.getN
      · simpa [raisedEnv, lowerEnv, setLocal] using ready.getBest
      · simpa [raisedEnv, lowerEnv, setLocal] using ready.getBestPhi
      · simp [raisedEnv, lowerEnv, setLocal]
      · simp [raisedEnv, lowerEnv, setLocal, pow_succ]
      · simpa [raisedEnv, lowerEnv, setLocal] using ready.getPhi
  · refine ⟨env, ?_, ?_⟩
    · apply BlockContinues.branch (flag := false)
        (by simpa only [boundary, decide_false] using tested) rfl
      exact .nil
    · simpa only [advanceWidth, if_neg boundary] using ready

end solvers.«70».scan
