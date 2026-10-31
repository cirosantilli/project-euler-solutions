import solvers.«89.py.lean.d».ParserLoop

namespace solvers.«89».roman_to_int
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false
set_option maxHeartbeats 800000

/-- The actual indexed Python parser terminates with the recursive model's exact value. -/
theorem result (text : String) (valid : RomanText text) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (table : SymbolTable heap globals) :
    ReturnsValue program "roman_to_int" [textValue text] heap
      (number (RomanNumerals.parse (text.toList.map String.singleton))) heap globals := by
  let args := ((findFunctionIn program.body "roman_to_int").getD ([], [])).1
  let source := ((findFunctionIn program.body "roman_to_int").getD ([], [])).2
  let entry := functionEnv args source [textValue text] globals
  have getText : entry "s" = some (textValue text) := by cases globals <;> rfl
  have getTable : entry "ROMAN_VALUE" = globals.lookup "ROMAN_VALUE" := by
    cases globals with
    | nil => rfl
    | cons item rest =>
        have notArgument : "ROMAN_VALUE" ∉ args.map Arg.name := by decide
        have notLocal : "ROMAN_VALUE" ∉ blockLocals source := by decide
        simp only [entry, functionEnv, notArgument, notLocal, if_false]
  let initialized := setLocal (setLocal (setLocal entry "total" (number 0)) "i" (number 0))
    "n" (number text.length)
  have ready : Locals initialized globals text 0 0 := by
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · simpa [initialized, setLocal] using getText
    · simp [initialized, setLocal]
    · simp [initialized, setLocal]
    · simp [initialized, setLocal]
    · simpa [initialized, setLocal] using getTable
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := number 0) Evaluates.constant
  apply BlockReturns.assign (value := number 0) Evaluates.constant
  apply BlockReturns.assign (value := number text.length) (middle := heap)
  · apply Evaluates.builtin (values := [textValue text]) (middle := heap)
    · exact .cons (Evaluates.lookup (by simpa [setLocal] using getText)) .nil
    · rfl
  simpa only [suffix, List.drop_zero, Nat.zero_add] using loop text valid 0 0 heap globals table initialized ready

end solvers.«89».roman_to_int
