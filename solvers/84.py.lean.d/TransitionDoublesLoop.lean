import solvers.«84.py.lean.d».TransitionCollectRow

namespace solvers.«84».transitions
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary
set_option Elab.async false

def doublesLoop : Stmt := .forIn (.name "doubles") (.call (.name "range") [.constant (.int 3)]) rowBody []
def positionStates (pos : Nat) : List (Nat × Nat) := (List.range' 0 3).map fun doubles => (pos, doubles)

theorem doubles_source : transition_row.loopBody (executableBody outcomes.definition.2) 5 = [doublesLoop] := rfl

theorem doubles_scan (sides : Nat) (positive : 0 < sides) (base : Mutable.Heap)
    (source tableRoot pos count first stop : Nat) (length : first + count = stop)
    (sourceView : outcomes.View base source (outcomes.all sides)) (table : transition_row.Table base tableRoot)
    (garbage : Mutable.Heap) (states : List (Nat × Nat)) (rows : List Row) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (views : Views sides (working base rows garbage) states rows) (ready : Ready env source tableRoot base.length)
    (getPos : env "pos" = some (number pos)) :
    ∃ (finalRows : List Row) (finalGarbage : Mutable.Heap) (finalEnv : HeapCalls.Env),
      BlockContinues program env [.rangeStepLoop "doubles" first stop 1 rowBody []]
        (working base rows garbage) finalEnv (working base finalRows finalGarbage) globals ∧
      Views sides (working base finalRows finalGarbage)
        (states ++ (List.range' first count).map fun doubles => (pos, doubles)) finalRows ∧
      Ready finalEnv source tableRoot base.length ∧ finalEnv "pos" = some (number pos) := by
  induction count generalizing first garbage states rows env with
  | zero =>
      refine ⟨rows, garbage, env, ?_, by simpa using views, ready, getPos⟩
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
  | succ count ih =>
      let bound := setLocal env "doubles" (number first)
      have boundReady : Ready bound source tableRoot base.length := by
        exact ⟨by simpa [bound, setLocal] using ready.getSource,
          by simpa [bound, setLocal] using ready.getTable,
          by simpa [bound, setLocal] using ready.getCollector⟩
      obtain ⟨row, extra, nextEnv, collected, nextViews, nextReady, nextPos⟩ :=
        collect_row sides positive base garbage states rows bound globals source tableRoot pos first sourceView table views boundReady
          (by simpa [bound, setLocal] using getPos) (by simp [bound, setLocal])
      obtain ⟨finalRows, finalGarbage, finalEnv, continued, finalViews, finalReady, finalPos⟩ :=
        ih (first + 1) (by omega) (garbage ++ extra) (states ++ [(pos, first)]) (rows ++ [row]) nextEnv nextViews nextReady nextPos
      refine ⟨finalRows, finalGarbage, finalEnv, ?_, ?_, finalReady, finalPos⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [show Calls.loopContinuation rowBody
          (.rangeStepLoop "doubles" ((first : Int) + 1) stop 1 rowBody []) =
          .rangeStepLoop "doubles" ((first : Int) + 1) stop 1 rowBody [] from rfl,
          Int.natCast_add, Int.natCast_one] using collected.append continued
      · simpa only [List.range'_succ, List.map_cons, List.append_assoc, List.singleton_append] using finalViews

theorem doubles_result (sides : Nat) (positive : 0 < sides) (base garbage : Mutable.Heap)
    (states : List (Nat × Nat)) (rows : List Row) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (source tableRoot pos : Nat)
    (sourceView : outcomes.View base source (outcomes.all sides)) (table : transition_row.Table base tableRoot)
    (views : Views sides (working base rows garbage) states rows) (ready : Ready env source tableRoot base.length)
    (getPos : env "pos" = some (number pos)) :
    ∃ (finalRows : List Row) (finalGarbage : Mutable.Heap) (finalEnv : HeapCalls.Env),
      BlockContinues program env [doublesLoop] (working base rows garbage) finalEnv (working base finalRows finalGarbage) globals ∧
      Views sides (working base finalRows finalGarbage) (states ++ positionStates pos) finalRows ∧
      Ready finalEnv source tableRoot base.length := by
  obtain ⟨finalRows, finalGarbage, finalEnv, ran, finalViews, finalReady, _⟩ :=
    doubles_scan sides positive base source tableRoot pos 3 0 3 rfl sourceView table garbage states rows env globals views ready getPos
  refine ⟨finalRows, finalGarbage, finalEnv, ?_, finalViews, finalReady⟩
  exact BlockContinues.for_range (.cons Evaluates.constant .nil) rfl ran

end solvers.«84».transitions
