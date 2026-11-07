import solvers.«98.py.lean.d».PatternTable

/-! The Python pattern helper returns the canonical first-occurrence labels of every string. -/
namespace solvers.«98».pattern_signature
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.AnagramSquares
set_option Elab.async false

abbrev prepareCode : Stmt := .ifThen (.membership (.name "ch") (.name "mp") true) recordCode []
abbrev appendCode : Stmt := .expr (.call (.attribute (.name "out") "append") [.subscript (.name "mp") (.name "ch")])
abbrev body : List Stmt := [prepareCode, appendCode]

theorem prepare (base : Mutable.Heap) (known : List Char) (output : List Nat) (c : Char)
    (env : HeapCalls.Env) (locals : Locals env base known) (getC : env "ch" = some (character c)) :
    ∃ afterEnv, BlockContinues program env [prepareCode] (work base known output) afterEnv
      (work base (extend known c) output) ∧ Locals afterEnv base (extend known c) ∧
      afterEnv "ch" = some (character c) := by
  have tested := membership base known output c env locals getC
  by_cases member : c ∈ known
  · simp only [extend, if_pos member]
    refine ⟨env, ?_, locals, getC⟩
    apply BlockContinues.branch (flag := false) (by simpa only [member, not_true_eq_false, decide_false] using tested) rfl
    exact .nil
  · simp only [extend, if_neg member]
    refine ⟨setLocal env "nxt" (number (known.length + 1)), ?_, ?_, ?_⟩
    · apply BlockContinues.branch (flag := true) (by simpa only [member, not_false_eq_true, decide_true] using tested) rfl
      exact record base known output c member env locals getC
    · exact ⟨by simpa [setLocal] using locals.getTable,
        by simpa [setLocal] using locals.getOutput, by simp [setLocal]⟩
    · simpa [setLocal] using getC

theorem loop (text : String) (remaining known : List Char) (output : List Nat) (base : Mutable.Heap)
    (position : Nat) (env : HeapCalls.Env) (suffix : text.toList.drop position = remaining)
    (locals : Locals env base known) :
    ∃ finalEnv finalKnown, BlockContinues program env
      [.sequenceStepLoop "ch" (.scalar (.string text)) position body []] (work base known output)
      finalEnv (work base finalKnown (output ++ labels known remaining)) ∧ Locals finalEnv base finalKnown := by
  induction remaining generalizing position known output env with
  | nil =>
      refine ⟨env, known, ?_, locals⟩
      simp only [labels, List.append_nil]
      apply BlockContinues.sequence_done (source := .scalar (.string text))
        (items := text.toList.map character) rfl
      · have missing : text.toList[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := text.toList) (i := position)).symm
        simp [missing]
      · exact .nil
  | cons c rest ih =>
      let entered := setLocal env "ch" (character c)
      have enteredLocals : Locals entered base known :=
        ⟨by simpa [entered, setLocal] using locals.getTable,
         by simpa [entered, setLocal] using locals.getOutput,
         by simpa [entered, setLocal] using locals.getNext⟩
      obtain ⟨prepared, preparedRun, preparedLocals, preparedC⟩ :=
        prepare base known output c entered enteredLocals (by simp [entered, setLocal])
      have appended := BlockContinues.discard (append base (extend known c) output c (mem_extend known c)
        prepared preparedLocals preparedC)
      rw [extend_idxOf] at appended
      obtain ⟨finalEnv, finalKnown, tailRun, finalLocals⟩ := ih (extend known c)
        (output ++ [known.idxOf c]) (position + 1) prepared
        (by simp [List.drop_add_one_eq_tail_drop, suffix]) preparedLocals
      refine ⟨finalEnv, finalKnown, ?_, finalLocals⟩
      have item : text.toList[position]? = some c := by
        simpa [suffix] using (List.head?_drop (l := text.toList) (i := position)).symm
      apply BlockContinues.sequence_step (source := .scalar (.string text))
        (items := text.toList.map character) (value := character c)
        rfl (by simp [item])
      simpa only [body, prepareCode, recordCode, appendCode, Calls.loopContinuation, Calls.loopControl,
        Calls.loopControlStmt, Bool.false_eq_true, if_false, labels, List.cons_append, List.nil_append,
        List.append_assoc, List.singleton_append] using preparedRun.append (appended.append tailRun)

theorem result (text : String) (heap : Mutable.Heap) :
    ∃ after address, ReturnsValue program "pattern_signature" [.scalar (.string text)] heap (.ref address) after ∧
      after[address]? = some (.tuple ((labels [] text.toList).map number)) ∧ heap <+: after := by
  let definition := (findFunctionIn program.body "pattern_signature").getD ([], [])
  let entry := functionEnv definition.1 definition.2 [.scalar (.string text)] []
  let withTable := setLocal entry "mp" (.ref heap.length)
  let withOutput := setLocal withTable "out" (.ref (heap.length + 1))
  let ready := setLocal withOutput "nxt" (number 0)
  obtain ⟨finalEnv, finalKnown, executed, finalLocals⟩ := loop text text.toList [] [] heap 0 ready (by simp)
    ⟨rfl, rfl, rfl⟩
  simp only [List.nil_append] at executed
  obtain ⟨address, extra, allocated, found⟩ := ProjectEulerSolutions.HeapCollection.allocate_tuple
    ((labels [] text.toList).map number) (work heap finalKnown (labels [] text.toList))
  have preserved : heap <+: work heap finalKnown (labels [] text.toList) ++ extra := by
    exact ⟨[.dict (table finalKnown 0), .list ((labels [] text.toList).map number)] ++ extra,
      by simp only [work, List.append_assoc]⟩
  refine ⟨_, address, ?_, found, preserved⟩
  apply BlockReturns.function rfl rfl
  apply (BlockContinues.annAssign (name := "mp") (Evaluates.dictionary .nil)).returns
  apply (BlockContinues.annAssign (name := "out") (Evaluates.list .nil)).returns
  apply BlockReturns.assign (value := number 0) Evaluates.constant
  apply BlockContinues.returns (body := [.forIn (.name "ch") (.name "s") body []])
    (middleEnv := finalEnv) (middleHeap := work heap finalKnown (labels [] text.toList))
  · have scanned : BlockContinues program ready [.forIn (.name "ch") (.name "s") body []]
        (work heap [] []) finalEnv (work heap finalKnown (labels [] text.toList)) := by
      apply BlockContinues.for_sequence (value := .scalar (.string text)) (middle := work heap [] [])
        (items := text.toList.map character) (by intros; simp)
        (Evaluates.lookup (show ready "s" = some (.scalar (.string text)) by rfl)) rfl
      exact executed
    simpa only [ready, withOutput, withTable, work, table, List.map_nil, List.append_assoc,
      List.length_append, List.length_singleton] using scanned
  · apply BlockReturns.return_value
    apply Evaluates.builtin (.cons (Evaluates.lookup finalLocals.getOutput) .nil)
    rw [primitive_tuple_list _ _ _ (found_output heap finalKnown (labels [] text.toList))]
    exact allocated

end solvers.«98».pattern_signature
