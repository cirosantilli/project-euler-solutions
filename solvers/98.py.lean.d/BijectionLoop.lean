import solvers.«98.py.lean.d».BijectionStep
import lang.py.FormalSW.ZipCalls

namespace solvers.«98».bijection
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.AnagramSquares ProjectEulerSolutions.AnagramMapping
open pattern_signature (program)
open data.«project-euler-statements».data.py.project_euler_statements.p98.mapping (value table)
set_option Elab.async false

abbrev target : Expr := .tuple [.name "ch", .name "d"]
abbrev unpack : Stmt := .assign [target] (.name "$formalsw.unpack")
abbrev sources (word square : String) : List SequenceSource := [.scalar (.string word), .scalar (.string square)]
abbrev cursor (word square : String) (position : Nat) : Stmt :=
  .zipStepLoop "$formalsw.unpack" (sources word square) position (unpack :: body) []
abbrev bound (env : HeapCalls.Env) (heap : Mutable.Heap) (letter digit : Char) : HeapCalls.Env :=
  assignNames (setLocal env "$formalsw.unpack" (.ref heap.length)) ["ch", "d"] [value letter, value digit]

def Keeps (env finalEnv : HeapCalls.Env) : Prop :=
  ∀ name, name ∉ ["$formalsw.unpack", "ch", "d"] → finalEnv name = env name

theorem unpack_result (env : HeapCalls.Env) (heap : Mutable.Heap) (letter digit : Char) :
    BlockContinues program (setLocal env "$formalsw.unpack" (.ref heap.length)) [unpack]
      (heap ++ [.tuple [value letter, value digit]]) (bound env heap letter digit)
      (heap ++ [.tuple [value letter, value digit]]) := by
  apply BlockContinues.unpack (names := ["ch", "d"]) (items := [value letter, value digit]) (value := .ref heap.length) rfl
    (Evaluates.lookup (by simp [setLocal]))
  · simp [Mutable.sequence]
  · rfl

/-- Every zip iteration succeeds for equal canonical patterns, and the completed tables encode the bijection. -/
theorem loop (remaining valueRest : List Char) (before valueBefore : List Char) (word square : String)
    (base garbage : Mutable.Heap) (letters digits : List Char) (env : HeapCalls.Env)
    (sameBefore : before.length = valueBefore.length)
    (wordView : word.toList = before ++ remaining) (squareView : square.toList = valueBefore ++ valueRest)
    (getForward : env "char_to_digit" = some (.ref base.length))
    (getReverse : env "digit_to_char" = some (.ref (base.length + 1)))
    (sameKnown : letters.length = digits.length) (uniqueLetters : letters.Nodup) (uniqueDigits : digits.Nodup)
    (matched : labels letters remaining = labels digits valueRest) :
    ∃ finalEnv extra, BlockContinues program env [cursor word square before.length]
      (work base letters digits garbage) finalEnv
      (work base (alphabet letters remaining) (alphabet digits valueRest) (garbage ++ extra)) ∧ Keeps env finalEnv := by
  induction remaining generalizing valueRest before valueBefore garbage letters digits env with
  | nil =>
      have empty : valueRest = [] := List.eq_nil_of_length_eq_zero (by simpa using congrArg List.length matched.symm)
      subst valueRest
      refine ⟨env, [], ?_, ?_⟩
      · simp only [alphabet, List.append_nil]
        apply BlockContinues.zip_done (rest := BlockContinues.nil)
        simp [zipItem, zipFields, sequenceValue, Mutable.sequence, wordView]
      · intro name absent; rfl
  | cons letter rest ih =>
      cases valueRest with
      | nil => simp [labels] at matched
      | cons digit tail =>
          have head : letters.idxOf letter = digits.idxOf digit := (List.cons.inj matched).1
          have tails : labels (extend letters letter) rest = labels (extend digits digit) tail := (List.cons.inj matched).2
          let heap := work base letters digits garbage
          let pair := Mutable.Object.tuple [value letter, value digit]
          let nextEnv := bound env heap letter digit
          have heapAppended : heap ++ [pair] = work base letters digits (garbage ++ [pair]) := by
            simp [heap, work, List.append_assoc]
          have locals : Locals nextEnv base letter digit := by
            constructor <;> simp_all [nextEnv, assignNames, setLocal]
          have kept : Keeps env nextEnv := by
            intro name absent
            simp only [List.mem_cons, not_or] at absent
            simp [nextEnv, assignNames, setLocal, absent]
          have item : zipItem (sources word square) before.length heap = some (.ok (some [value letter, value digit]), heap) := by
            simp [zipItem, zipFields, sequenceValue, Mutable.sequence, wordView, squareView, sameBefore]
          have allocate : Mutable.allocateTuple [value letter, value digit] heap = some (.ok (.ref heap.length), heap ++ [pair]) := rfl
          dsimp only [heap] at item allocate
          have prepared := unpack_result env heap letter digit
          have checked := step_success base (garbage ++ [pair]) letters digits letter digit nextEnv locals sameKnown uniqueLetters uniqueDigits head
          obtain ⟨finalEnv, extra, continued, preserved⟩ :=
            ih tail (before ++ [letter]) (valueBefore ++ [digit]) (garbage ++ [pair])
              (extend letters letter) (extend digits digit) nextEnv
              (by simpa using sameBefore)
              (by simpa only [List.append_assoc, List.singleton_append] using wordView)
              (by simpa only [List.append_assoc, List.singleton_append] using squareView)
              locals.forward locals.reverse (extend_lengths letters digits letter digit sameKnown head)
              (extend_nodup letters letter uniqueLetters) (extend_nodup digits digit uniqueDigits) tails
          refine ⟨finalEnv, pair :: extra, ?_, ?_⟩
          · intro following anchors stack
            refine Reaches.next ?_ ((prepared (body ++ [.loopResume (cursor word square (before.length + 1))] ++ following) anchors stack).trans ?_)
            · simp [step, cursor, run, item, allocate]
              exact ⟨⟨rfl, rfl⟩, rfl, rfl⟩
            rw [heapAppended]
            apply (checked (.loopResume (cursor word square (before.length + 1)) :: following) anchors stack).trans
            refine Reaches.next (following := ⟨.block (cursor word square (before.length + 1) :: following) nextEnv,
              stack, work base (extend letters letter) (extend digits digit) (garbage ++ [pair])⟩) ?_ ?_
            · rfl
            simpa only [alphabet, List.length_append, List.length_singleton, List.append_assoc, List.singleton_append] using
              continued following anchors stack
          · intro name absent
            exact (preserved name absent).trans (kept name absent)

abbrev prepareCode : List Stmt := candidateBody.take 3
abbrev loopCode : Stmt := .forZip target [.name "w1", .name "sq_str"] body []
abbrev initializedEnv (env : HeapCalls.Env) (heap : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal (setLocal env "char_to_digit" (.ref heap.length)) "digit_to_char" (.ref (heap.length + 1)))
    "ok" (.scalar (.bool true))

theorem source_checker : candidateBody.take 4 = prepareCode ++ [loopCode] := rfl

theorem initialized (env : HeapCalls.Env) (heap : Mutable.Heap) :
    BlockContinues program env prepareCode heap (initializedEnv env heap) (work heap [] [] []) := by
  apply BlockContinues.append (body := prepareCode.take 1)
    (BlockContinues.annAssign (Evaluates.dictionary .nil))
  apply BlockContinues.append (body := (prepareCode.drop 1).take 1)
    (BlockContinues.annAssign (Evaluates.dictionary .nil))
  convert
    (BlockContinues.assign (module := program) (globals := [])
      (env := setLocal (setLocal env "char_to_digit" (.ref heap.length)) "digit_to_char" (.ref (heap.length + 1)))
      (heap := (heap ++ [.dict []]) ++ [.dict []]) (name := "ok")
      (Evaluates.constant (value := .bool true))) using 1 <;>
    simp [work, table, List.append_assoc]

/-- The actual mapping preparation and complete zip loop terminate with mutually inverse tables and ok still true. -/
theorem result (word square : String) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (matched : labels [] word.toList = labels [] square.toList)
    (getWord : env "w1" = some (.scalar (.string word)))
    (getSquare : env "sq_str" = some (.scalar (.string square))) :
    ∃ finalEnv garbage,
      BlockContinues program env (candidateBody.take 4) heap finalEnv
        (work heap (alphabet [] word.toList) (alphabet [] square.toList) garbage) ∧
      finalEnv "char_to_digit" = some (.ref heap.length) ∧
      finalEnv "digit_to_char" = some (.ref (heap.length + 1)) ∧
      finalEnv "ok" = some (.scalar (.bool true)) ∧
      (∀ name, name ∉ ["char_to_digit", "digit_to_char", "ok", "$formalsw.unpack", "ch", "d"] → finalEnv name = env name) := by
  obtain ⟨finalEnv, garbage, checked, kept⟩ := loop word.toList square.toList [] [] word square heap [] [] []
    (initializedEnv env heap) rfl rfl rfl (by simp [initializedEnv, setLocal]) (by simp [initializedEnv, setLocal])
    rfl (by simp) (by simp) matched
  refine ⟨finalEnv, garbage, ?_, ?_, ?_, ?_, ?_⟩
  · rw [source_checker]
    apply (initialized env heap).append
    apply BlockContinues.for_zip
      (.cons (Evaluates.lookup (show (initializedEnv env heap) "w1" = some (.scalar (.string word)) by
        simpa [initializedEnv, setLocal] using getWord))
        (.cons (Evaluates.lookup (show (initializedEnv env heap) "sq_str" = some (.scalar (.string square)) by
          simpa [initializedEnv, setLocal] using getSquare)) .nil))
    · rfl
    · exact checked
  · rw [kept "char_to_digit" (by decide)]
    simp [initializedEnv, setLocal]
  · rw [kept "digit_to_char" (by decide)]
    simp [initializedEnv, setLocal]
  · rw [kept "ok" (by decide)]
    simp [initializedEnv, setLocal]
  · intro name absent
    rw [kept name (by intro member; exact absent (by simp_all))]
    simp only [List.mem_cons, not_or] at absent
    simp [initializedEnv, setLocal, absent]

end solvers.«98».bijection
