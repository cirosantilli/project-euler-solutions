import «.formalsw».solvers.«89.py.ast»
import ProjectEulerSolutions.Lib.RomanNumerals
import lang.py.FormalSW.DictionaryCalls

namespace solvers.«89»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

abbrev program := Solvers.P89Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
abbrev textValue (text : String) : Mutable.Value := .scalar (.string text)
abbrev character (c : Char) : Mutable.Value := textValue (String.singleton c)

def RomanText (text : String) : Prop := ∀ c ∈ text.toList, String.singleton c ∈ RomanNumerals.symbols

/-- Only the Roman symbol keys are observed; other dictionary entries are allowed. -/
def SymbolTable (heap : Mutable.Heap) (globals : Mutable.Globals) : Prop :=
  ∃ address entries,
    globals.lookup "ROMAN_VALUE" = some (.ref address) ∧
    heap[address]? = some (.dict entries) ∧
    ∀ symbol ∈ RomanNumerals.symbols,
    ScalarDict.lookup entries (.string symbol) = some (number (RomanNumerals.value symbol))

namespace roman_to_int
abbrev code : List Stmt := executableBody ((findFunctionIn program.body "roman_to_int").getD ([], [])).2
abbrev body : List Stmt := match code[3]'(by decide) with
  | .whileDo _ body _ => body
  | _ => []

structure Locals (env : HeapCalls.Env) (globals : Mutable.Globals) (text : String) (index total : Nat) : Prop where
  input : env "s" = some (textValue text)
  length : env "n" = some (number text.length)
  position : env "i" = some (number index)
  total : env "total" = some (number total)
  table : env "ROMAN_VALUE" = globals.lookup "ROMAN_VALUE"

theorem Locals.value {env : HeapCalls.Env} {globals : Mutable.Globals} {text : String} {index total : Nat}
    (ready : Locals env globals text index total) (value : Nat) :
    Locals (setLocal env "v" (number value)) globals text index total := by
  rcases ready with ⟨input, length, position, sum, table⟩
  exact ⟨by simpa [setLocal] using input, by simpa [setLocal] using length,
    by simpa [setLocal] using position, by simpa [setLocal] using sum, by simpa [setLocal] using table⟩

theorem Locals.second {env : HeapCalls.Env} {globals : Mutable.Globals} {text : String} {index total : Nat}
    (ready : Locals env globals text index total) (value : Nat) :
    Locals (setLocal env "v2" (number value)) globals text index total := by
  rcases ready with ⟨input, length, position, sum, table⟩
  exact ⟨by simpa [setLocal] using input, by simpa [setLocal] using length,
    by simpa [setLocal] using position, by simpa [setLocal] using sum, by simpa [setLocal] using table⟩

theorem Locals.advance {env : HeapCalls.Env} {globals : Mutable.Globals} {text : String} {index total : Nat}
    (ready : Locals env globals text index total) (next newTotal : Nat) :
    Locals (setLocal (setLocal env "total" (number newTotal)) "i" (number next)) globals text next newTotal := by
  rcases ready with ⟨input, length, _, _, table⟩
  exact ⟨by simpa [setLocal] using input, by simpa [setLocal] using length,
    by simp [setLocal], by simp [setLocal], by simpa [setLocal] using table⟩

theorem string_item (text : String) (index : Nat) (inside : index < text.toList.length) (heap : Mutable.Heap) :
    Mutable.getItem (textValue text) (number index) heap = some (.ok (character text.toList[index]), heap) := by
  have positive : ¬ (index : Int) < 0 := by omega
  have bounded : ¬ text.length ≤ index := by simpa using Nat.not_le.mpr inside
  simp [Mutable.getItem, Strings.get, FormalSW.Py.Builtin.Sequence.get?,
    FormalSW.Py.Builtin.Sequence.normalizeIndex, textValue, character, positive, bounded,
    List.getElem?_eq_getElem inside]

theorem read (text : String) (index : Nat) (inside : index < text.toList.length) (valid : RomanText text)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (table : SymbolTable heap globals)
    (env : HeapCalls.Env) (getText : env "s" = some (textValue text))
    (getTable : env "ROMAN_VALUE" = globals.lookup "ROMAN_VALUE") (expression : Expr)
    (ordinary : ∀ a b c, expression ≠ .slice a b c)
    (position : Evaluates program env expression heap (number index) heap globals) :
    Evaluates program env (.subscript (.name "ROMAN_VALUE") (.subscript (.name "s") expression)) heap
      (number (RomanNumerals.value (String.singleton text.toList[index]))) heap globals := by
  obtain ⟨address, entries, bound, found, correct⟩ := table
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup (getTable.trans bound))
  · exact Evaluates.subscript ordinary
      (Evaluates.lookup getText) position (string_item text index inside heap)
  · exact Mutable.getItem_dictionary heap address entries found _
      (correct _ (valid _ (List.getElem_mem inside)))

theorem active (text : String) (index total : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (env : HeapCalls.Env) (ready : Locals env globals text index total) :
    Evaluates program env (.compare (.name "i") [(.lt, .name "n")]) heap
      (.scalar (.bool (decide (index < text.toList.length)))) heap globals := by
  apply Evaluates.compare (Evaluates.lookup ready.position) (Evaluates.lookup ready.length)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem next_position (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (index : Nat) (position : env "i" = some (number index)) :
    Evaluates program env (.binOp (.name "i") .add (.constant (.int 1))) heap
      (number (index + 1)) heap globals := by
  exact Evaluates.binary (Evaluates.lookup position) Evaluates.constant
    (by simp [Mutable.binary, scalarBinary, number])

theorem has_second (text : String) (index total : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (env : HeapCalls.Env) (ready : Locals env globals text index total) :
    Evaluates program env (.compare (.binOp (.name "i") .add (.constant (.int 1))) [(.lt, .name "n")]) heap
      (.scalar (.bool (decide (index + 1 < text.toList.length)))) heap globals := by
  apply Evaluates.compare (next_position env heap globals index ready.position) (Evaluates.lookup ready.length)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  omega

theorem increasing (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (first second : Nat) (getFirst : env "v" = some (number first))
    (getSecond : env "v2" = some (number second)) :
    Evaluates program env (.compare (.name "v") [(.lt, .name "v2")]) heap
      (.scalar (.bool (decide (first < second)))) heap globals := by
  apply Evaluates.compare (Evaluates.lookup getFirst) (Evaluates.lookup getSecond)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

end roman_to_int
end solvers.«89»
