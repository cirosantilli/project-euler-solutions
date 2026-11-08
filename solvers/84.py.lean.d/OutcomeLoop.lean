import solvers.«84.py.lean.d».OutcomeIteration

namespace solvers.«84».outcomes
open FormalSW.PyAst HeapCalls
set_option Elab.async false

/-- All sums are visited in source order, with ordinary and double categories kept separate. -/
theorem loop (sides : Nat) (base : Mutable.Heap) (globals : Mutable.Globals)
    (length first stop : Nat) (endAt : first + length = stop)
    (values : List Outcome) (env : HeapCalls.Env)
    (getSides : env "dice_sides" = some (number sides)) (getOutcomes : env "outcomes" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "move" first stop 1 code []]
      (heap base values) finalEnv (heap base (values ++ (List.range' first length).flatMap (fun move : Nat => perMove sides move))) globals ∧
      finalEnv "dice_sides" = some (number sides) ∧ finalEnv "outcomes" = some (.ref base.length) := by
  induction length generalizing first values env with
  | zero =>
      refine ⟨env, ?_, getSides, getOutcomes⟩
      simp only [List.range'_zero, List.flatMap_nil, List.append_nil]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ length ih =>
      let entered := setLocal env "move" (number first)
      have enteredSides : entered "dice_sides" = some (number sides) := by simpa [entered, setLocal] using getSides
      have enteredOutcomes : entered "outcomes" = some (.ref base.length) := by simpa [entered, setLocal] using getOutcomes
      have visited := iteration sides first base values entered globals enteredSides (by simp [entered, setLocal]) enteredOutcomes
      have updatedSides : locals entered sides first "dice_sides" = some (number sides) := by simpa [locals, setLocal] using enteredSides
      have updatedOutcomes : locals entered sides first "outcomes" = some (.ref base.length) := by simpa [locals, setLocal] using enteredOutcomes
      obtain ⟨finalEnv, continued, finalSides, finalOutcomes⟩ := ih (first + 1) (by omega)
        (values ++ perMove sides first) (locals entered sides first) updatedSides updatedOutcomes
      refine ⟨finalEnv, ?_, finalSides, finalOutcomes⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, code, ordinaryCode, doubleCode,
        Bool.false_or, Bool.false_eq_true, if_false, List.range'_succ, List.flatMap_cons, List.append_assoc,
        Int.natCast_add, Int.natCast_one] using visited.append continued

def all (sides : Nat) : List Outcome := (List.range' 2 (2 * sides - 1)).flatMap fun move : Nat => perMove sides move

theorem collect (sides : Nat) (positive : 0 < sides) (base : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getSides : env "dice_sides" = some (number sides)) :
    ∃ finalEnv, BlockContinues program env
      [.assign [.name "outcomes"] (.list []),
        .forIn (.name "move") (.call (.name "range") [.constant (.int 2),
          .binOp (.binOp (.constant (.int 2)) .mult (.name "dice_sides")) .add (.constant (.int 1))]) code []]
      base finalEnv (heap base (all sides)) globals ∧
      finalEnv "dice_sides" = some (number sides) ∧ finalEnv "outcomes" = some (.ref base.length) := by
  let initialized := setLocal env "outcomes" (.ref base.length)
  have initialSides : initialized "dice_sides" = some (number sides) := by simpa [initialized, setLocal] using getSides
  obtain ⟨finalEnv, collected, finalSides, finalOutcomes⟩ := loop sides base globals (2 * sides - 1) 2 (2 * sides + 1)
    (by omega) [] initialized initialSides (by simp [initialized, setLocal])
  refine ⟨finalEnv, ?_, finalSides, finalOutcomes⟩
  have allocate := BlockContinues.assign (name := "outcomes") (Evaluates.list (module := program) (env := env) (heap := base)
    (globals := globals) Evaluates.Arguments.nil)
  have range : BlockContinues program initialized
      [.forIn (.name "move") (.call (.name "range") [.constant (.int 2),
        .binOp (.binOp (.constant (.int 2)) .mult (.name "dice_sides")) .add (.constant (.int 1))]) code []]
      (heap base []) finalEnv (heap base (all sides)) globals := by
    apply BlockContinues.for_range (start := 2) (stop := (2 * sides + 1 : Nat)) (stride := 1)
      (.cons Evaluates.constant (.cons
        (add initialized (heap base []) globals _ _ (2 * sides) 1
          (mul initialized (heap base []) globals _ _ 2 sides Evaluates.constant (Evaluates.lookup initialSides)) Evaluates.constant) .nil))
    · change Except.ok (2, 2 * (sides : Int) + 1, 1) = Except.ok (2, ((2 * sides + 1 : Nat) : Int), 1)
      simp
    · simpa [all] using collected
  simpa only [List.cons_append, List.nil_append, heap, references, List.length_nil, List.range_zero, List.map_nil] using allocate.append range

end solvers.«84».outcomes
