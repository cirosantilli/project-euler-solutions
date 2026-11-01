import solvers.«92.py.lean.d».Transition
import lang.py.FormalSW.IntegerParsing

namespace solvers.«92».prefixStep
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := transition.program
abbrev body : List Stmt :=
  .assign [.name "ceiling"] (.builtinCall .int [.name "character"]) :: transition.body

abbrev digit := Prefix.characterValue

theorem source :
    (match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[2]? with
      | some (Stmt.forIn _ _ prefixBody _) => prefixBody
      | _ => []) = body := rfl

theorem parse (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (character : Char) (valid : character.isDigit)
    (lookup : before "character" = some (.scalar (.string (String.singleton character)))) :
    Evaluates program before (.builtinCall .int [.name "character"]) heap
      (table.number (digit character)) heap globals := by
  apply Evaluates.builtin (.cons (Evaluates.lookup lookup) .nil)
  simp [primitive, Calls.integerConversion_digit character valid, scalarResult,
    digit, Prefix.characterValue, table.number, next_term.number]

def rebound : List String := "ceiling" :: transition.rebound

def LocalsFrame (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ∉ rebound → after name = before name

/-- Parse the current printed digit, then build exactly the next prefix distribution. -/
theorem result (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (address : Nat) (entries : table.Entries) (character : Char) (valid : character.isDigit)
    (lookup : before "character" = some (.scalar (.string (String.singleton character))))
    (counts : before "counts" = some (.ref address))
    (found : table.Found heap address entries) (keys : table.Keys heap entries) :
    ∃ result extra finalLocals,
      BlockContinues program before body heap finalLocals (heap ++ extra) globals ∧
      table.Found (heap ++ extra) heap.length result ∧ table.Keys (heap ++ extra) result ∧
      table.view result = Prefix.transition (digit character) (table.view entries) ∧
      finalLocals "counts" = some (.ref heap.length) ∧ LocalsFrame before finalLocals := by
  let initial := setLocal before "ceiling" (table.number (digit character))
  obtain ⟨result, extra, finalLocals, ran, finalFound, finalKeys, view, getCounts, localFrame⟩ :=
    transition.result heap initial globals address (digit character) entries
      (by simpa [initial, setLocal] using counts) (by simp [initial, setLocal]) found keys
  refine ⟨result, extra, finalLocals, ?_, finalFound, finalKeys, view, getCounts, ?_⟩
  · exact (BlockContinues.assign (parse heap before globals character valid lookup)).append ran
  · intro name keep
    simp only [rebound, List.mem_cons, not_or] at keep
    rw [localFrame name keep.2]
    simp [initial, setLocal, keep.1]

end solvers.«92».prefixStep
