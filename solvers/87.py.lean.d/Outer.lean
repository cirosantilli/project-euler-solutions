import solvers.«87.py.lean.d».Middle

namespace solvers.«87».search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PrimePowerTriples
set_option Elab.async false

abbrev outerBody : List Stmt := [.forIn (.name "c") (.name "cubes") middleBody []]

theorem outer_loop (limit : Nat) (squares cubes fourths rest : List Nat) (position : Nat)
    (suffix : fourths.drop position = rest) (base : Mutable.Heap) (addresses : Addresses)
    (storedSquares : base[addresses.squares]? = some (.list (encode squares)))
    (storedCubes : base[addresses.cubes]? = some (.list (encode cubes)))
    (storedFourths : base[addresses.fourths]? = some (.list (encode fourths)))
    (before : List Nat) (env : HeapCalls.Env) (ready : Locals env limit addresses base.length) :
    ∃ finalEnv, BlockContinues program env [.sequenceStepLoop "f" (.ref addresses.fourths) position outerBody []]
      (working base before) finalEnv (working base (before ++ prunedValues limit squares cubes rest)) ∧
      Locals finalEnv limit addresses base.length := by
  induction rest generalizing position before env with
  | nil =>
      refine ⟨env, ?_, ready⟩
      simp only [prunedValues, List.flatMap_nil, List.append_nil]
      apply BlockContinues.sequence_done (source := .ref addresses.fourths) (items := encode fourths)
        (sequence base before fourths addresses.fourths storedFourths)
      · have missing : fourths[position]? = none := by simpa [suffix] using (List.head?_drop (l := fourths) (i := position)).symm
        simp [encode, missing]
      · exact .nil
  | cons head tail ih =>
      let entered := setLocal env "f" (number head)
      have enteredReady := ready.set_other "f" (number head) (by decide) (by decide) (by decide) (by decide) (by decide)
      obtain ⟨nextEnv, rowExecuted, nextReady, _⟩ := middle_result limit head squares cubes base addresses
        storedSquares storedCubes before entered enteredReady (by simp [entered, setLocal])
      obtain ⟨finalEnv, continued, finalReady⟩ := ih (position + 1)
        (by simp [List.drop_add_one_eq_tail_drop, suffix]) (before ++ middleValues limit head squares cubes) nextEnv nextReady
      refine ⟨finalEnv, ?_, finalReady⟩
      have selected : fourths[position]? = some head := by simpa [suffix] using (List.head?_drop (l := fourths) (i := position)).symm
      apply BlockContinues.sequence_step (source := .ref addresses.fourths) (items := encode fourths)
        (value := number head) (sequence base before fourths addresses.fourths storedFourths) (by simp [encode, selected])
      simpa only [Calls.loopContinuation, show Calls.loopControl outerBody = false from rfl,
        Bool.false_eq_true, if_false, prunedValues, List.flatMap_cons, middleValues, List.append_assoc] using
        rowExecuted.append continued

/-- The actual pruned loops insert precisely the visited sums into their fresh set. -/
theorem result (limit : Nat) (squares cubes fourths : List Nat) (base : Mutable.Heap) (addresses : Addresses)
    (storedSquares : base[addresses.squares]? = some (.list (encode squares)))
    (storedCubes : base[addresses.cubes]? = some (.list (encode cubes)))
    (storedFourths : base[addresses.fourths]? = some (.list (encode fourths)))
    (before : List Nat) (env : HeapCalls.Env) (ready : Locals env limit addresses base.length) :
    ∃ finalEnv, BlockContinues program env [.forIn (.name "f") (.name "fourths") outerBody []]
      (working base before) finalEnv (working base (before ++ prunedValues limit squares cubes fourths)) ∧
      Locals finalEnv limit addresses base.length := by
  obtain ⟨finalEnv, executed, finalReady⟩ := outer_loop limit squares cubes fourths fourths 0 rfl base addresses
    storedSquares storedCubes storedFourths before env ready
  refine ⟨finalEnv, ?_, finalReady⟩
  apply BlockContinues.for_sequence (value := .ref addresses.fourths) (items := encode fourths)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup ready.fourths)
    (sequence base before fourths addresses.fourths storedFourths)
  exact executed

end solvers.«87».search
