import solvers.«89.py.lean.d».EncoderExecution
import solvers.«89.py.lean.d».ParserExecution

namespace solvers.«89»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions RomanNumerals
set_option Elab.async false
set_option maxHeartbeats 1000000

/-- Existing global tables survive the encoder's private allocations. -/
theorem SymbolTable.append {heap : Mutable.Heap} {globals : Mutable.Globals}
    (table : SymbolTable heap globals) (extra : Mutable.Heap) : SymbolTable (heap ++ extra) globals := by
  obtain ⟨address, entries, bound, found, correct⟩ := table
  exact ⟨address, entries, bound, HeapCollection.prefix_found heap extra address _ found, correct⟩

theorem EncodingTable.append {heap : Mutable.Heap} {globals : Mutable.Globals}
    (table : EncodingTable heap globals) (extra : Mutable.Heap) : EncodingTable (heap ++ extra) globals := by
  obtain ⟨address, entries, bound, found, valid, models⟩ := table
  refine ⟨address, entries, bound, HeapCollection.prefix_found heap extra address _ found, ?_, models⟩
  intro entry member
  exact ⟨(valid entry member).positive, HeapCollection.prefix_found heap extra entry.address _ (valid entry member).found⟩

def ValidText (text : String) : Prop := text.toList ≠ [] ∧ RomanText text
abbrev decoded (text : String) : Nat := parse (text.toList.map String.singleton)
abbrev minimal (text : String) : String := String.ofList (canonical (decoded text))
def saved (text : String) : Int := (text.length : Int) - (canonical (decoded text)).length

theorem decoded_positive (text : String) (valid : ValidText text) : 0 < decoded text := by
  apply parse_positive
  · simpa using valid.1
  · intro symbol member
    obtain ⟨c, inChars, rfl⟩ := List.mem_map.mp member
    exact valid.2 c inChars

namespace solve
abbrev code : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
abbrev body : List Stmt := match code[1]'(by decide) with
  | .forIn _ _ body _ => body
  | _ => []

/-- Parse and encode one input line, then add the signed difference of its lengths. -/
theorem iteration (text : String) (valid : ValidText text) (total : Int)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (symbols : SymbolTable heap globals)
    (tokens : EncodingTable heap globals) (env : HeapCalls.Env)
    (getText : env "s" = some (textValue text)) (getTotal : env "saved" = some (.scalar (.int total))) :
    ∃ extra finalEnv, BlockContinues program env body heap finalEnv (heap ++ extra) globals ∧
      finalEnv "saved" = some (.scalar (.int (total + saved text))) := by
  let parsed := setLocal env "val" (number (decoded text))
  let encoded := setLocal parsed "minimal" (textValue (minimal text))
  let summed := setLocal encoded "saved" (.scalar (.int (total + saved text)))
  have first : BlockContinues program env [body[0]'(by decide)] heap parsed heap globals :=
    BlockContinues.assign (Evaluates.call (.cons (Evaluates.lookup getText) .nil)
      (roman_to_int.result text valid.2 heap globals symbols))
  obtain ⟨extra, rendered⟩ := int_to_min_roman.result (decoded text) (decoded_positive text valid) heap globals tokens
  have second : BlockContinues program parsed [body[1]'(by decide)] heap encoded (heap ++ extra) globals :=
    BlockContinues.assign (Evaluates.call (.cons (Evaluates.lookup (by simp [parsed, setLocal])) .nil) rendered)
  have third : BlockContinues program encoded [body[2]'(by decide)] (heap ++ extra) summed (heap ++ extra) globals := by
    apply BlockContinues.augAssign (middle := heap ++ extra) (left := .scalar (.int total)) (right := .scalar (.int (saved text)))
    · simpa [encoded, parsed, setLocal] using getTotal
    · apply Evaluates.binary (av := number text.length) (bv := number (minimal text).length)
        (middle := heap ++ extra) (afterRight := heap ++ extra)
      · apply Evaluates.builtin (values := [textValue text]) (middle := heap ++ extra)
        · exact .cons (Evaluates.lookup (by simpa [encoded, parsed, setLocal] using getText)) .nil
        · rfl
      · apply Evaluates.builtin (values := [textValue (minimal text)]) (middle := heap ++ extra)
        · exact .cons (Evaluates.lookup (by simp [encoded, setLocal])) .nil
        · rfl
      · simp [Mutable.binary, scalarBinary, saved, minimal]
    · rfl
  exact ⟨extra, summed, first.append (second.append third), by simp [summed, setLocal]⟩

end solve
end solvers.«89»
