import solvers.«92.py.lean.d».PrefixStep

namespace solvers.«92».prefixLoop
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := prefixStep.program
abbrev characterValue (character : Char) : Mutable.Value := .scalar (.string (String.singleton character))

def cursor (text : String) (position : Nat) : Stmt :=
  .sequenceStepLoop "character" (.scalar (.string text)) position prefixStep.body []

def rebound : List String := "character" :: prefixStep.rebound

def LocalsFrame (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ∉ rebound → after name = before name

/-- Traverse the remaining decimal characters, preserving all original heap objects. -/
theorem scan (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (text : String) (valid : ∀ character ∈ text.toList, character.isDigit)
    (position address : Nat) (entries : table.Entries)
    (counts : before "counts" = some (.ref address))
    (found : table.Found heap address entries) (keys : table.Keys heap entries) :
    ∃ result extra finalAddress finalLocals,
      BlockContinues program before [cursor text position] heap finalLocals (heap ++ extra) globals ∧
      table.Found (heap ++ extra) finalAddress result ∧ table.Keys (heap ++ extra) result ∧
      table.view result = (text.toList.drop position).foldl
        (fun table character => Prefix.transition (prefixStep.digit character) table) (table.view entries) ∧
      finalLocals "counts" = some (.ref finalAddress) ∧ LocalsFrame before finalLocals := by
  have sequence : Mutable.sequence (sequenceValue (.scalar (.string text))) heap =
      some (.ok (text.toList.map characterValue), heap) := by simp [sequenceValue, Mutable.sequence]
  by_cases active : position < text.toList.length
  · let character := text.toList[position]
    let initial := setLocal before "character" (characterValue character)
    obtain ⟨updated, firstExtra, locals, digitRan, middleFound, middleKeys, digitView, middleCounts, digitLocals⟩ :=
      prefixStep.result heap initial globals address entries character
        (valid character (List.getElem_mem active))
        (by simp [initial, setLocal]) (by simpa [initial, setLocal] using counts) found keys
    have rowLocals : LocalsFrame before locals := by
      intro name keep
      simp only [rebound, List.mem_cons, not_or] at keep
      rw [digitLocals name keep.2]
      simp [initial, setLocal, keep.1]
    obtain ⟨result, restExtra, finalAddress, finalLocals, rest, finalFound, finalKeys, restView,
      finalCounts, localFrame⟩ :=
      scan (heap ++ firstExtra) locals globals text valid (position + 1) heap.length updated
        middleCounts middleFound middleKeys
    refine ⟨result, firstExtra ++ restExtra, finalAddress, finalLocals, ?_, ?_, ?_, ?_, finalCounts, ?_⟩
    · rw [← List.append_assoc]
      apply BlockContinues.sequence_step sequence
        (value := characterValue character)
        (by simp [List.getElem?_map, List.getElem?_eq_getElem active, character])
      simpa only [show Calls.loopContinuation prefixStep.body (cursor text (position + 1)) =
        cursor text (position + 1) from rfl] using digitRan.append rest
    · simpa only [List.append_assoc] using finalFound
    · simpa only [List.append_assoc] using finalKeys
    · rw [restView, digitView, List.drop_eq_getElem_cons active]
      rfl
    · intro name keep
      exact (localFrame name keep).trans (rowLocals name keep)
  · have exhausted : text.toList.length ≤ position := by omega
    refine ⟨entries, [], address, before, ?_, ?_, ?_, ?_, counts, ?_⟩
    · simpa only [List.append_nil] using
        (BlockContinues.sequence_done (body := prefixStep.body) sequence
          (by simp [List.getElem?_map, List.getElem?_eq_none exhausted]) BlockContinues.nil)
    · simpa using found
    · simpa using keys
    · simp [List.drop_eq_nil_of_le exhausted]
    · intro name _
      rfl
termination_by text.toList.length - position
decreasing_by omega

abbrev iterable : Expr := .builtinCall .str
  [.binOp (.name "limit_exclusive") .sub (.constant (.int 1))]
abbrev loop : Stmt := .forIn (.name "character") iterable prefixStep.body []

theorem source :
    (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[2]? = some loop := rfl

theorem printed (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (bound : Nat) (limit : before "limit_exclusive" = some (table.number (bound + 1))) :
    Evaluates program before iterable heap (.scalar (.string (toString (bound : Int)))) heap globals := by
  apply Evaluates.builtin (values := [table.number bound]) (middle := heap)
  · apply Evaluates.Arguments.cons _ .nil
    apply Evaluates.binary (Evaluates.lookup limit) Evaluates.constant
    simp [Mutable.binary, scalarBinary, table.number, next_term.number]
  · rfl

/-- The actual Python string loop produces the exact distribution for the exclusive bound minus one. -/
theorem result (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (bound address : Nat) (entries : table.Entries)
    (limit : before "limit_exclusive" = some (table.number (bound + 1)))
    (counts : before "counts" = some (.ref address))
    (found : table.Found heap address entries) (keys : table.Keys heap entries)
    (initial : Prefix.Exact 0 (table.view entries)) :
    ∃ result extra finalAddress finalLocals,
      BlockContinues program before [loop] heap finalLocals (heap ++ extra) globals ∧
      table.Found (heap ++ extra) finalAddress result ∧ table.Keys (heap ++ extra) result ∧
      Prefix.Exact bound (table.view result) ∧
      finalLocals "counts" = some (.ref finalAddress) ∧ LocalsFrame before finalLocals := by
  let text := toString (bound : Int)
  have characters : text.toList = Nat.toDigits 10 bound := by simp [text, Int.repr, Nat.repr]
  have valid : ∀ character ∈ text.toList, character.isDigit := by
    intro character present
    exact Nat.isDigit_of_mem_toDigits (b := 10) (by decide) (by decide) (by simpa only [characters] using present)
  obtain ⟨result, extra, finalAddress, finalLocals, ran, finalFound, finalKeys, view, finalCounts, localFrame⟩ :=
    scan heap before globals text valid 0 address entries counts found keys
  refine ⟨result, extra, finalAddress, finalLocals, ?_, finalFound, finalKeys, ?_, finalCounts, localFrame⟩
  · apply BlockContinues.for_sequence (items := text.toList.map characterValue)
      (by intro expressions impossible; cases impossible)
      (printed heap before globals bound limit)
      (by simp [Mutable.sequence, text])
    exact ran
  · rw [view]
    have represented := initial.fold_characters text.toList valid
    simpa only [characters, Nat.ofDigitChars_ten_toDigits, List.drop_zero] using represented

end solvers.«92».prefixLoop
