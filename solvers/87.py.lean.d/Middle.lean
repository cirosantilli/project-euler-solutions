import solvers.«87.py.lean.d».Inner

namespace solvers.«87».search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PrimePowerTriples
set_option Elab.async false

abbrev middleBody : List Stmt :=
  [.assign [.name "fc"] (.binOp (.name "f") .add (.name "c")),
   .ifThen (.compare (.name "fc") [(.gtE, .name "limit")]) [.breakLoop] [],
   .forIn (.name "s") (.name "squares") innerBody []]

def middleValues (limit f : Nat) (squares cubes : List Nat) : List Nat :=
  (cubes.takeWhile (fun c => decide (f + c < limit))).flatMap (fun c => rowValues limit (f + c) squares)

theorem middle_loop (limit f : Nat) (squares cubes rest : List Nat) (position : Nat)
    (suffix : cubes.drop position = rest) (base : Mutable.Heap) (addresses : Addresses)
    (storedSquares : base[addresses.squares]? = some (.list (encode squares)))
    (storedCubes : base[addresses.cubes]? = some (.list (encode cubes)))
    (before : List Nat) (env : HeapCalls.Env) (ready : Locals env limit addresses base.length)
    (getF : env "f" = some (number f)) :
    ∃ finalEnv, BlockContinues program env [.sequenceStepLoop "c" (.ref addresses.cubes) position middleBody []]
      (working base before) finalEnv (working base (before ++ middleValues limit f squares rest)) ∧
      Locals finalEnv limit addresses base.length ∧ finalEnv "f" = some (number f) := by
  induction rest generalizing position before env with
  | nil =>
      refine ⟨env, ?_, ready, getF⟩
      simp only [middleValues, List.takeWhile_nil, List.flatMap_nil, List.append_nil]
      apply BlockContinues.sequence_done (source := .ref addresses.cubes) (items := encode cubes)
        (sequence base before cubes addresses.cubes storedCubes)
      · have missing : cubes[position]? = none := by simpa [suffix] using (List.head?_drop (l := cubes) (i := position)).symm
        simp [encode, missing]
      · exact .nil
  | cons head tail ih =>
      let entered := setLocal env "c" (number head)
      let measured := setLocal entered "fc" (number (f + head))
      have enteredReady := ready.set_other "c" (number head) (by decide) (by decide) (by decide) (by decide) (by decide)
      have measuredReady := enteredReady.set_other "fc" (number (f + head)) (by decide) (by decide) (by decide) (by decide) (by decide)
      have measuredF : measured "f" = some (number f) := by simpa [measured, entered, setLocal] using getF
      have measuredOffset : measured "fc" = some (number (f + head)) := by simp [measured, setLocal]
      have assigned : BlockContinues program entered [middleBody[0]'(by decide)] (working base before) measured (working base before) := by
        apply BlockContinues.assign
        exact Evaluates.binary (av := number f) (bv := number head)
          (Evaluates.lookup (by simpa [entered, setLocal] using getF)) (Evaluates.lookup (by simp [entered, setLocal]))
          (by simp [Mutable.binary, scalarBinary, number])
      have tested := threshold limit (f + head) (working base before) measured "fc" measuredReady.bound measuredOffset
      have selected : cubes[position]? = some head := by simpa [suffix] using (List.head?_drop (l := cubes) (i := position)).symm
      by_cases stop : limit ≤ f + head
      · refine ⟨measured, ?_, measuredReady, measuredF⟩
        simp only [middleValues, List.takeWhile_cons, show decide (f + head < limit) = false by simp; omega,
          Bool.false_eq_true, if_false, List.flatMap_nil, List.append_nil]
        apply BlockContinues.sequence_step (source := .ref addresses.cubes) (items := encode cubes)
          (value := number head) (sequence base before cubes addresses.cubes storedCubes) (by simp [encode, selected])
        change BlockContinues program entered
          ([middleBody[0]'(by decide)] ++
            [middleBody[1]'(by decide), middleBody[2]'(by decide),
             .loopResume (.sequenceStepLoop "c" (.ref addresses.cubes) (position + 1) middleBody [])])
          (working base before) measured (working base before)
        apply assigned.append
        apply BlockContinues.branch_next (flag := true) tested (by simp [Mutable.truth, scalarTruth, stop])
        exact BlockContinues.break_loop (rest := []) rfl .nil
      · have active : f + head < limit := by omega
        have guarded : BlockContinues program measured [middleBody[1]'(by decide)]
            (working base before) measured (working base before) :=
          BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, stop]) .nil
        obtain ⟨rowEnv, rowExecuted, rowReady, _, rowF⟩ := inner_result limit (f + head) squares base addresses
          storedSquares before measured measuredReady measuredOffset
        obtain ⟨finalEnv, continued, finalReady, finalF⟩ := ih (position + 1)
          (by simp [List.drop_add_one_eq_tail_drop, suffix]) (before ++ rowValues limit (f + head) squares)
          rowEnv rowReady (rowF.trans measuredF)
        refine ⟨finalEnv, ?_, finalReady, finalF⟩
        apply BlockContinues.sequence_step (source := .ref addresses.cubes) (items := encode cubes)
          (value := number head) (sequence base before cubes addresses.cubes storedCubes) (by simp [encode, selected])
        have resumed := BlockContinues.loop_resume continued
        simpa only [Calls.loopContinuation, show Calls.loopControl middleBody = true from rfl, if_true,
          middleValues, List.takeWhile_cons, show decide (f + head < limit) = true by simp [active],
          List.flatMap_cons, List.append_assoc] using assigned.append (guarded.append (rowExecuted.append resumed))

theorem middle_result (limit f : Nat) (squares cubes : List Nat) (base : Mutable.Heap) (addresses : Addresses)
    (storedSquares : base[addresses.squares]? = some (.list (encode squares)))
    (storedCubes : base[addresses.cubes]? = some (.list (encode cubes)))
    (before : List Nat) (env : HeapCalls.Env) (ready : Locals env limit addresses base.length)
    (getF : env "f" = some (number f)) :
    ∃ finalEnv, BlockContinues program env [.forIn (.name "c") (.name "cubes") middleBody []]
      (working base before) finalEnv (working base (before ++ middleValues limit f squares cubes)) ∧
      Locals finalEnv limit addresses base.length ∧ finalEnv "f" = some (number f) := by
  obtain ⟨finalEnv, executed, finalReady, finalF⟩ := middle_loop limit f squares cubes cubes 0 rfl base addresses
    storedSquares storedCubes before env ready getF
  refine ⟨finalEnv, ?_, finalReady, finalF⟩
  apply BlockContinues.for_sequence (value := .ref addresses.cubes) (items := encode cubes)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup ready.cubes)
    (sequence base before cubes addresses.cubes storedCubes)
  exact executed

end solvers.«87».search
