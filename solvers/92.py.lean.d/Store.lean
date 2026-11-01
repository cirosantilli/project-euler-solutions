import solvers.«92.py.lean.d».Digits
import lang.py.FormalSW.NaturalDictionaryCalls
import ProjectEulerSolutions.Lib.DigitSquareMemo

namespace solvers.«92».store
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := Solvers.P92Ast.module
abbrev number := NaturalDictionary.number
abbrev Entries := NaturalDictionary.Entries
abbrev encode := NaturalDictionary.encode
abbrev lookup := NaturalDictionary.lookup
abbrev insert := NaturalDictionary.insert
abbrev Found := NaturalDictionary.Found

theorem lookup_insert (entries : Entries) (key value : Nat) :
    lookup (insert entries key value) = Memo.write (lookup entries) key value :=
  NaturalDictionary.lookup_insert_function entries key value

def fill (entries : Entries) (value : Nat) : List Nat → Entries
  | [] => entries
  | first :: rest => fill (insert entries first value) value rest

theorem fill_lookup (entries : Entries) (value : Nat) (path : List Nat) :
    lookup (fill entries value path) = Memo.fill (lookup entries) path value := by
  induction path generalizing entries with
  | nil => rfl
  | cons head tail ih =>
      simpa only [fill, Memo.fill, List.foldl_cons, lookup_insert] using ih (insert entries head value)

theorem fill_valid (entries : Entries) (value : Nat) (path : List Nat)
    (valid : Memo.Valid (lookup entries))
    (correct : ∀ n ∈ path, decide (value = 89) = endsAt89 n) :
    Memo.Valid (lookup (fill entries value path)) := by
  rw [fill_lookup]
  exact Memo.fill_valid (lookup entries) path value valid correct

def initial : Entries := [(0, 0), (1, 1), (89, 89)]

theorem initial_lookup : lookup initial = Memo.initial := by
  funext n
  simp [initial, lookup, NaturalDictionary.lookup, Memo.initial]

theorem initial_valid : Memo.Valid (lookup initial) := by
  rw [initial_lookup]
  exact Memo.initial_valid

/-- The solver's three endpoint entries are created by the actual dictionary display. -/
theorem initial_result (heap : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals) :
    Evaluates program locals
      (.dict [(.constant (.int 0), .constant (.int 0)),
        (.constant (.int 1), .constant (.int 1)), (.constant (.int 89), .constant (.int 89))])
      heap (.ref heap.length) (heap ++ [.dict (encode initial)]) globals := by
  exact Evaluates.dictionary_constants [(.int 0, .int 0), (.int 1, .int 1), (.int 89, .int 89)]

end solvers.«92».store
