import solvers.«84.py.lean.d».TransitionDoublesLoop

namespace solvers.«84».transitions
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary
set_option Elab.async false

def positionLoop : Stmt := .forIn (.name "pos") (.call (.name "range") [.constant (.int 40)]) [doublesLoop] []
def allStates : List (Nat × Nat) := (List.range' 0 40).flatMap positionStates

theorem position_source : (executableBody outcomes.definition.2)[(5 : Nat)]? = some positionLoop := rfl

theorem position_scan (sides : Nat) (positive : 0 < sides) (base : Mutable.Heap)
    (source tableRoot count first stop : Nat) (length : first + count = stop)
    (sourceView : outcomes.View base source (outcomes.all sides)) (table : transition_row.Table base tableRoot)
    (garbage : Mutable.Heap) (states : List (Nat × Nat)) (rows : List Row) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (views : Views sides (working base rows garbage) states rows) (ready : Ready env source tableRoot base.length) :
    ∃ (finalRows : List Row) (finalGarbage : Mutable.Heap) (finalEnv : HeapCalls.Env),
      BlockContinues program env [.rangeStepLoop "pos" first stop 1 [doublesLoop] []]
        (working base rows garbage) finalEnv (working base finalRows finalGarbage) globals ∧
      Views sides (working base finalRows finalGarbage)
        (states ++ (List.range' first count).flatMap positionStates) finalRows ∧
      Ready finalEnv source tableRoot base.length := by
  induction count generalizing first garbage states rows env with
  | zero =>
      refine ⟨rows, garbage, env, ?_, by simpa using views, ready⟩
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
  | succ count ih =>
      let bound := setLocal env "pos" (number first)
      have boundReady : Ready bound source tableRoot base.length :=
        ⟨by simpa [bound, setLocal] using ready.getSource,
          by simpa [bound, setLocal] using ready.getTable,
          by simpa [bound, setLocal] using ready.getCollector⟩
      obtain ⟨nextRows, nextGarbage, nextEnv, collected, nextViews, nextReady⟩ :=
        doubles_result sides positive base garbage states rows bound globals source tableRoot first sourceView table views boundReady
          (by simp [bound, setLocal])
      obtain ⟨finalRows, finalGarbage, finalEnv, continued, finalViews, finalReady⟩ :=
        ih (first + 1) (by omega) nextGarbage (states ++ positionStates first) nextRows nextEnv nextViews nextReady
      refine ⟨finalRows, finalGarbage, finalEnv, ?_, ?_, finalReady⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [show Calls.loopContinuation [doublesLoop]
          (.rangeStepLoop "pos" ((first : Int) + 1) stop 1 [doublesLoop] []) =
          .rangeStepLoop "pos" ((first : Int) + 1) stop 1 [doublesLoop] [] from rfl,
          Int.natCast_add, Int.natCast_one] using collected.append continued
      · simpa only [List.range'_succ, List.flatMap_cons, List.append_assoc] using finalViews

/-- Every square contributes its three consecutive doubles states in source order. -/
theorem position_result (sides : Nat) (positive : 0 < sides) (base garbage : Mutable.Heap)
    (states : List (Nat × Nat)) (rows : List Row) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (source tableRoot : Nat)
    (sourceView : outcomes.View base source (outcomes.all sides)) (table : transition_row.Table base tableRoot)
    (views : Views sides (working base rows garbage) states rows) (ready : Ready env source tableRoot base.length) :
    ∃ (finalRows : List Row) (finalGarbage : Mutable.Heap) (finalEnv : HeapCalls.Env),
      BlockContinues program env [positionLoop] (working base rows garbage) finalEnv (working base finalRows finalGarbage) globals ∧
      Views sides (working base finalRows finalGarbage) (states ++ allStates) finalRows ∧
      Ready finalEnv source tableRoot base.length := by
  obtain ⟨finalRows, finalGarbage, finalEnv, ran, finalViews, finalReady⟩ :=
    position_scan sides positive base source tableRoot 40 0 40 rfl sourceView table garbage states rows env globals views ready
  exact ⟨finalRows, finalGarbage, finalEnv, BlockContinues.for_range (.cons Evaluates.constant .nil) rfl ran,
    finalViews, finalReady⟩

end solvers.«84».transitions
