import solvers.«74.py.lean.d».Digits
import lang.py.FormalSW.DictionaryCalls
import ProjectEulerStatements.Lib.DigitFactorialMemo

/-! Natural-key dictionary views for the source cache and visited-position table. -/
namespace solvers.«74».store
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev program := Solvers.P74Ast.module
abbrev number := factorials.number
abbrev Entries := List (Nat × Nat)
abbrev encode (entries : Entries) : List (Constant × Mutable.Value) :=
  entries.map (fun entry => (.int entry.1, number entry.2))

def lookup : Entries → Nat → Option Nat
  | [], _ => none
  | (stored, value) :: rest, key => if key = stored then some value else lookup rest key

def insert : Entries → Nat → Nat → Entries
  | [], key, value => [(key, value)]
  | (stored, old) :: rest, key, value =>
      if key = stored then (stored, value) :: rest else (stored, old) :: insert rest key value

theorem lookup_encode (entries : Entries) (key : Nat) :
    ScalarDict.lookup (encode entries) (.int key) = (lookup entries key).map number := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      rcases entry with ⟨stored, value⟩
      by_cases same : key = stored
      · subst key
        simp [encode, ScalarDict.lookup, lookup, scalarCompare, evalCmpOp]
      · simp [encode, ScalarDict.lookup, lookup, scalarCompare, evalCmpOp, same,
          Ne.symm same, ih]

theorem insert_encode (entries : Entries) (key value : Nat) :
    ScalarDict.insert (encode entries) (.int key) (number value) =
      encode (insert entries key value) := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      rcases entry with ⟨stored, old⟩
      by_cases same : key = stored
      · subst key
        simp [encode, ScalarDict.insert, insert, scalarCompare, evalCmpOp]
      · simp [encode, ScalarDict.insert, insert, scalarCompare, evalCmpOp, same,
          Ne.symm same, ih]

theorem lookup_insert (entries : Entries) (key value query : Nat) :
    lookup (insert entries key value) query =
      if query = key then some value else lookup entries query := by
  induction entries with
  | nil => simp [insert, lookup]
  | cons entry rest ih =>
      rcases entry with ⟨stored, old⟩
      by_cases same : key = stored <;> by_cases here : query = stored <;>
        by_cases chosen : query = key <;> simp_all [insert, lookup]

theorem lookup_insert_function (entries : Entries) (key value : Nat) :
    lookup (insert entries key value) =
      ProjectEulerStatements.DigitFactorialChains.Memo.write (lookup entries) key value := by
  funext query
  exact lookup_insert entries key value query

abbrev Found (heap : Mutable.Heap) (address : Nat) (entries : Entries) : Prop :=
  heap[address]? = some (.dict (encode entries))

theorem found_inside {heap : Mutable.Heap} {address : Nat} {entries : Entries}
    (found : Found heap address entries) : address < heap.length :=
  (List.getElem?_eq_some_iff.mp found).1

theorem found_append {heap : Mutable.Heap} {address : Nat} {entries : Entries}
    (found : Found heap address entries) (extra : Mutable.Heap) :
    Found (heap ++ extra) address entries := by
  exact (List.getElem?_append_left (found_inside found)).trans found

theorem found_set {heap : Mutable.Heap} {address : Nat} {entries : Entries}
    (found : Found heap address entries) (replacement : Entries) :
    Found (heap.set address (.dict (encode replacement))) address replacement := by
  exact List.getElem?_set_self (found_inside found)

theorem found_other {heap : Mutable.Heap} {address other : Nat} {entries : Entries}
    (found : Found heap address entries) (object : Mutable.Object) (different : other ≠ address) :
    Found (heap.set other object) address entries :=
  (List.getElem?_set_ne different).trans found

variable {globals : Mutable.Globals}

theorem membership_result (heap : Mutable.Heap) (address : Nat) (entries : Entries)
    (found : Found heap address entries) (key : Nat) (negated : Bool) :
    HeapCalls.membership (number key) (.ref address) negated heap =
      some (.ok (.scalar (.bool (if negated then !(lookup entries key).isSome
        else (lookup entries key).isSome))), heap) := by
  simp [HeapCalls.membership, found, lookup_encode, number, factorials.number]

theorem membership_evaluates (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (name keyName : String) (address key : Nat) (entries : Entries)
    (found : Found heap address entries) (getDict : locals name = some (.ref address))
    (getKey : locals keyName = some (number key)) (negated : Bool) :
    Evaluates program locals (.membership (.name keyName) (.name name) negated) heap
      (.scalar (.bool (if negated then !(lookup entries key).isSome
        else (lookup entries key).isSome))) heap globals :=
  Evaluates.membership (.cons (Evaluates.lookup getKey) (.cons (Evaluates.lookup getDict) .nil))
    (membership_result heap address entries found key negated)

theorem lookup_evaluates (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (name keyName : String) (address key value : Nat) (entries : Entries)
    (found : Found heap address entries) (getDict : locals name = some (.ref address))
    (getKey : locals keyName = some (number key)) (selected : lookup entries key = some value) :
    Evaluates program locals (.subscript (.name name) (.name keyName)) heap (number value) heap
      globals := by
  apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup getDict)
    (Evaluates.lookup getKey)
  exact Mutable.getItem_dictionary heap address (encode entries) found (.int key)
    (by rw [lookup_encode, selected]; rfl)

theorem insert_evaluates (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (name keyName : String) (address key value : Nat) (entries : Entries) (expression : Expr)
    (found : Found heap address entries) (getDict : locals name = some (.ref address))
    (getKey : locals keyName = some (number key))
    (evaluated : Evaluates program locals expression heap (number value) heap globals) :
    BlockContinues program locals [.assign [.subscript (.name name) (.name keyName)] expression]
      heap locals (heap.set address (.dict (encode (insert entries key value)))) globals := by
  apply BlockContinues.setItem (by intro a b c; simp) evaluated
    (.cons (Evaluates.lookup getDict) (.cons (Evaluates.lookup getKey) .nil))
  simpa only [insert_encode] using
    Mutable.setItem_dictionary heap address (encode entries) found (.int key) (number value)

end solvers.«74».store
