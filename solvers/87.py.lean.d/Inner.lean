import solvers.«87.py.lean.d».SetCollector

namespace solvers.«87».search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PrimePowerTriples
set_option Elab.async false

abbrev innerBody : List Stmt :=
  [.assign [.name "total"] (.binOp (.name "fc") .add (.name "s")),
   .ifThen (.compare (.name "total") [(.gtE, .name "limit")]) [.breakLoop] [],
   .expr (.call (.attribute (.name "representable") "add") [.name "total"])]

theorem inner_loop (limit offset : Nat) (squares rest : List Nat) (position : Nat)
    (suffix : squares.drop position = rest) (base : Mutable.Heap) (addresses : Addresses)
    (stored : base[addresses.squares]? = some (.list (encode squares)))
    (before : List Nat) (env : HeapCalls.Env) (ready : Locals env limit addresses base.length)
    (getOffset : env "fc" = some (number offset)) :
    ∃ finalEnv, BlockContinues program env [.sequenceStepLoop "s" (.ref addresses.squares) position innerBody []]
      (working base before) finalEnv (working base (before ++ rowValues limit offset rest)) ∧
      Locals finalEnv limit addresses base.length ∧ finalEnv "fc" = some (number offset) ∧
      finalEnv "f" = env "f" := by
  induction rest generalizing position before env with
  | nil =>
      refine ⟨env, ?_, ready, getOffset, rfl⟩
      simp only [rowValues, List.takeWhile_nil, List.map_nil, List.append_nil]
      apply BlockContinues.sequence_done (source := .ref addresses.squares) (items := encode squares)
        (sequence base before squares addresses.squares stored)
      · have missing : squares[position]? = none := by simpa [suffix] using (List.head?_drop (l := squares) (i := position)).symm
        simp [encode, missing]
      · exact .nil
  | cons head tail ih =>
      let entered := setLocal env "s" (number head)
      let measured := setLocal entered "total" (number (offset + head))
      have enteredReady := ready.set_other "s" (number head) (by decide) (by decide) (by decide) (by decide) (by decide)
      have measuredReady := enteredReady.set_other "total" (number (offset + head)) (by decide) (by decide) (by decide) (by decide) (by decide)
      have measuredOffset : measured "fc" = some (number offset) := by simpa [measured, entered, setLocal] using getOffset
      have measuredTotal : measured "total" = some (number (offset + head)) := by simp [measured, setLocal]
      have assigned : BlockContinues program entered [innerBody[0]'(by decide)] (working base before) measured (working base before) := by
        apply BlockContinues.assign
        exact Evaluates.binary (av := number offset) (bv := number head)
          (Evaluates.lookup (by simpa [entered, setLocal] using getOffset)) (Evaluates.lookup (by simp [entered, setLocal]))
          (by simp [Mutable.binary, scalarBinary, number])
      have tested := threshold limit (offset + head) (working base before) measured "total" measuredReady.bound measuredTotal
      have selected : squares[position]? = some head := by simpa [suffix] using (List.head?_drop (l := squares) (i := position)).symm
      by_cases stop : limit ≤ offset + head
      · refine ⟨measured, ?_, measuredReady, measuredOffset, by simp [measured, entered, setLocal]⟩
        simp only [rowValues, List.takeWhile_cons, show decide (offset + head < limit) = false by simp; omega,
          Bool.false_eq_true, if_false, List.map_nil, List.append_nil]
        apply BlockContinues.sequence_step (source := .ref addresses.squares) (items := encode squares)
          (value := number head) (sequence base before squares addresses.squares stored) (by simp [encode, selected])
        change BlockContinues program entered
          ([innerBody[0]'(by decide)] ++
            [innerBody[1]'(by decide), innerBody[2]'(by decide),
             .loopResume (.sequenceStepLoop "s" (.ref addresses.squares) (position + 1) innerBody [])])
          (working base before) measured (working base before)
        apply assigned.append
        apply BlockContinues.branch_next (flag := true) tested (by simp [Mutable.truth, scalarTruth, stop])
        exact BlockContinues.break_loop (rest := []) rfl .nil
      · have active : offset + head < limit := by omega
        have guarded : BlockContinues program measured [innerBody[1]'(by decide)]
            (working base before) measured (working base before) :=
          BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, stop]) .nil
        have inserted := add_value base before (offset + head) measured measuredReady.set measuredTotal
        obtain ⟨finalEnv, continued, finalReady, finalOffset, finalF⟩ := ih (position + 1)
          (by simp [List.drop_add_one_eq_tail_drop, suffix]) (before ++ [offset + head]) measured measuredReady measuredOffset
        refine ⟨finalEnv, ?_, finalReady, finalOffset, by simpa [measured, entered, setLocal] using finalF⟩
        apply BlockContinues.sequence_step (source := .ref addresses.squares) (items := encode squares)
          (value := number head) (sequence base before squares addresses.squares stored) (by simp [encode, selected])
        have resumed := BlockContinues.loop_resume continued
        simpa only [Calls.loopContinuation, show Calls.loopControl innerBody = true from rfl, if_true,
          rowValues, List.takeWhile_cons, show decide (offset + head < limit) = true by simp [active], if_true,
          List.map_cons, List.append_assoc, List.singleton_append] using
          assigned.append (guarded.append (inserted.append resumed))

theorem inner_result (limit offset : Nat) (squares : List Nat) (base : Mutable.Heap) (addresses : Addresses)
    (stored : base[addresses.squares]? = some (.list (encode squares)))
    (before : List Nat) (env : HeapCalls.Env) (ready : Locals env limit addresses base.length)
    (getOffset : env "fc" = some (number offset)) :
    ∃ finalEnv, BlockContinues program env [.forIn (.name "s") (.name "squares") innerBody []]
      (working base before) finalEnv (working base (before ++ rowValues limit offset squares)) ∧
      Locals finalEnv limit addresses base.length ∧ finalEnv "fc" = some (number offset) ∧ finalEnv "f" = env "f" := by
  obtain ⟨finalEnv, executed, finalReady, finalOffset, finalF⟩ := inner_loop limit offset squares squares 0 rfl base addresses stored before env ready getOffset
  refine ⟨finalEnv, ?_, finalReady, finalOffset, finalF⟩
  apply BlockContinues.for_sequence (value := .ref addresses.squares) (items := encode squares)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup ready.squares)
    (sequence base before squares addresses.squares stored)
  exact executed

end solvers.«87».search
