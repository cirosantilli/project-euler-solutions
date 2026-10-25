import lang.py.FormalSW.KeyDictionaryCalls
import Mathlib.Data.Int.Basic

/-! An ordered dictionary with integer-pair keys. The stored representative is
the actual tuple reference from the first insertion, even after later updates. -/
namespace ProjectEulerSolutions.IntPairCache
open FormalSW.PyAst

def key (left right : Int) : HashKey := .tuple [.scalar (.int left), .scalar (.int right)]

theorem key_equal (a b c d : Int) :
    HashKey.equal (key a b) (key c d) = decide (a = c ∧ b = d) := by
  simp [key, HashKey.equal, HashKey.depth, HashKey.listDepth, HashKey.equalFuel,
    List.isEqv, scalarCompare, evalCmpOp]

structure Entry where
  left : Int
  right : Int
  address : Nat
  answer : Bool

def Entry.encode (entry : Entry) : HashDict.Entry Mutable.Value :=
  (key entry.left entry.right, .ref entry.address, .scalar (.bool entry.answer))

def encode (entries : List Entry) : List (HashDict.Entry Mutable.Value) := entries.map Entry.encode

def lookup : List Entry → Int → Int → Option Bool
  | [], _, _ => none
  | entry :: rest, left, right =>
      if entry.left = left ∧ entry.right = right then some entry.answer else lookup rest left right

def insert : List Entry → Int → Int → Nat → Bool → List Entry
  | [], left, right, address, answer => [⟨left, right, address, answer⟩]
  | entry :: rest, left, right, address, answer =>
      if entry.left = left ∧ entry.right = right then { entry with answer } :: rest
      else entry :: insert rest left right address answer

theorem lookup_encode (entries : List Entry) (left right : Int) :
    HashDict.lookup (encode entries) (key left right) =
      (lookup entries left right).map (fun answer => .scalar (.bool answer)) := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      simp only [encode, List.map_cons, HashDict.lookup_cons, Entry.encode, key_equal]
      by_cases same : entry.left = left ∧ entry.right = right
      · simp [lookup, same]
      · simpa [encode, lookup, same] using ih

theorem insert_encode (entries : List Entry) (left right : Int) (address : Nat) (answer : Bool) :
    HashDict.insert (encode entries) (key left right) (.ref address) (.scalar (.bool answer)) =
      encode (insert entries left right address answer) := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      simp only [encode, List.map_cons, HashDict.insert, Entry.encode, key_equal]
      by_cases same : entry.left = left ∧ entry.right = right
      · simp [insert, same, Entry.encode]
      · simpa [insert, same, encode, Entry.encode] using congrArg (Entry.encode entry :: ·) ih

def Correct (relation : Int → Int → Bool) (entries : List Entry) : Prop :=
  ∀ entry ∈ entries, entry.answer = relation entry.left entry.right

variable {relation : Int → Int → Bool} {entries : List Entry}
  {left right : Int} {address : Nat} {answer : Bool}

theorem correct_nil (relation : Int → Int → Bool) : Correct relation [] := by
  intro entry member
  simp at member

theorem lookup_correct (correct : Correct relation entries)
    (found : lookup entries left right = some answer) : answer = relation left right := by
  induction entries with
  | nil => simp [lookup] at found
  | cons entry rest ih =>
      by_cases same : entry.left = left ∧ entry.right = right
      · have value : entry.answer = answer := by simpa [lookup, same] using found
        simpa [same.1, same.2, value] using correct entry (by simp)
      · apply ih (fun item member => correct item (by simp [member]))
        simpa [lookup, same] using found

theorem insert_correct (correct : Correct relation entries)
    (value : answer = relation left right) :
    Correct relation (insert entries left right address answer) := by
  induction entries with
  | nil =>
      intro entry member
      simp only [insert, List.mem_singleton] at member
      subst entry
      exact value
  | cons entry rest ih =>
      by_cases same : entry.left = left ∧ entry.right = right
      · intro item member
        simp only [insert, if_pos same, List.mem_cons] at member
        rcases member with rfl | member
        · simpa only [same.1, same.2] using value
        · exact correct item (by simp [member])
      · intro item member
        simp only [insert, if_neg same, List.mem_cons] at member
        rcases member with rfl | member
        · exact correct item (by simp)
        · exact ih (fun item present => correct item (by simp [present])) item member

def object : List Entry → Mutable.Object
  | [] => .dict []
  | entry :: rest => .keyDict (encode (entry :: rest))

theorem insert_ne_nil (entries : List Entry) : insert entries left right address answer ≠ [] := by
  cases entries with
  | nil => simp [insert]
  | cons entry rest => simp only [insert]; split <;> simp

theorem object_nonempty (nonempty : entries ≠ []) : object entries = .keyDict (encode entries) := by
  cases entries with
  | nil => contradiction
  | cons => rfl

theorem get (heap : Mutable.Heap) (address tuple : Nat) (entries : List Entry) (left right : Int)
    (found : heap[address]? = some (object entries))
    (tupleFound : heap[tuple]? = some (.tuple [.scalar (.int left), .scalar (.int right)])) :
    Mutable.dictionaryGet address (.ref tuple) (.scalar .none) heap =
      some (.ok (((lookup entries left right).map (fun b => .scalar (.bool b))).getD (.scalar .none)), heap) := by
  have hash : Mutable.hashKey (.ref tuple) heap = some (.ok (key left right), heap) :=
    Mutable.hashKey_tuple_scalars heap tuple [.int left, .int right] tupleFound
  cases entries with
  | nil =>
      simp [object, Mutable.dictionaryGet, found, hash, Mutable.promoteDictionary,
        HashDict.lookup, HashDict.find?, lookup, Except.map]
  | cons entry rest =>
      rw [Mutable.dictionaryGet_keyed heap address _ found _ _ hash, lookup_encode]

theorem set (heap : Mutable.Heap) (address tuple : Nat) (entries : List Entry) (left right : Int) (answer : Bool)
    (found : heap[address]? = some (object entries))
    (tupleFound : heap[tuple]? = some (.tuple [.scalar (.int left), .scalar (.int right)])) :
    Mutable.setItem (.ref address) (.ref tuple) (.scalar (.bool answer)) heap =
      some (.ok (.scalar .none), heap.set address (object (insert entries left right tuple answer))) := by
  have hash : Mutable.hashKey (.ref tuple) heap = some (.ok (key left right), heap) :=
    Mutable.hashKey_tuple_scalars heap tuple [.int left, .int right] tupleFound
  rw [object_nonempty (insert_ne_nil entries)]
  cases entries with
  | nil =>
      simp [object, Mutable.setItem, found, hash, Mutable.promoteDictionary, HashDict.insert,
        encode, Entry.encode, insert, Except.map]
  | cons entry rest =>
      simp only [object] at found
      simp [Mutable.setItem, found, hash, insert_encode, Except.map]

end ProjectEulerSolutions.IntPairCache
