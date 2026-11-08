import solvers.«84.py.lean.d».LandingPool
import ProjectEulerSolutions.Lib.TupleDictionary

namespace solvers.«84».resolve_landing.weights
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

structure Entry where
  landing : Int × Bool
  address : Nat
  weight : Int

def encode (entries : List Entry) : List (HashDict.Entry Mutable.Value) :=
  entries.map fun entry => (.tuple (pool.pair entry.landing.1 entry.landing.2), .ref entry.address, .scalar (.int entry.weight))

def lookup : List Entry → Int × Bool → Option Int
  | [], _ => none
  | entry :: rest, landing => if entry.landing = landing then some entry.weight else lookup rest landing

def insert : List Entry → Int × Bool → Nat → Int → List Entry
  | [], landing, address, weight => [⟨landing, address, weight⟩]
  | entry :: rest, landing, address, weight =>
      if entry.landing = landing then { entry with weight := weight } :: rest
      else entry :: insert rest landing address weight

def add (entries : List Entry) (landing : Int × Bool) (address : Nat) (weight : Int) : List Entry :=
  insert entries landing address ((lookup entries landing).getD 0 + weight)

theorem key_equal (left right : Int × Bool) :
    HashKey.equal (.tuple (pool.pair left.1 left.2)) (.tuple (pool.pair right.1 right.2)) = decide (left = right) := by
  rcases left with ⟨a, flag⟩
  rcases right with ⟨b, other⟩
  cases flag <;> cases other <;>
    simp [pool.pair, HashKey.equal, HashKey.depth, HashKey.listDepth, HashKey.equalFuel, List.isEqv, scalarCompare, evalCmpOp]

theorem lookup_encode (entries : List Entry) (landing : Int × Bool) :
    HashDict.lookup (encode entries) (.tuple (pool.pair landing.1 landing.2)) =
      (lookup entries landing).map (fun weight => .scalar (.int weight)) := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      simp only [encode] at ih
      by_cases same : entry.landing = landing <;> simp [encode, HashDict.lookup_cons, key_equal, lookup, same, ih]

theorem insert_encode (entries : List Entry) (landing : Int × Bool) (address : Nat) (weight : Int) :
    HashDict.insert (encode entries) (.tuple (pool.pair landing.1 landing.2)) (.ref address) (.scalar (.int weight)) =
      encode (insert entries landing address weight) := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      simp only [encode] at ih
      by_cases same : entry.landing = landing <;> simp [encode, HashDict.insert, key_equal, insert, same, ih]

theorem lookup_insert (entries : List Entry) (landing query : Int × Bool) (address : Nat) (weight : Int) :
    lookup (insert entries landing address weight) query = if query = landing then some weight else lookup entries query := by
  induction entries with
  | nil => by_cases same : landing = query <;> simp_all [insert, lookup, eq_comm]
  | cons entry rest ih =>
      by_cases same : entry.landing = landing <;> by_cases here : entry.landing = query <;>
        by_cases chosen : query = landing <;> simp_all [insert, lookup]

theorem lookup_add (entries : List Entry) (landing query : Int × Bool) (address : Nat) (weight : Int) :
    (lookup (add entries landing address weight) query).getD 0 =
      (lookup entries query).getD 0 + if query = landing then weight else 0 := by
  simp only [add, lookup_insert]
  by_cases same : query = landing <;> simp [same]

def Valid (heap : Mutable.Heap) (entries : List Entry) : Prop :=
  ∀ entry ∈ entries, TuplePool.View heap (.tuple (pool.pair entry.landing.1 entry.landing.2)) (.ref entry.address)

theorem Valid.append {heap : Mutable.Heap} {entries : List Entry} (valid : Valid heap entries) (extra : Mutable.Heap) :
    Valid (heap ++ extra) entries := fun entry member => (valid entry member).append extra

theorem Valid.inserted {heap : Mutable.Heap} {entries : List Entry} (valid : Valid heap entries)
    (landing : Int × Bool) (address : Nat) (weight : Int)
    (key : TuplePool.View heap (.tuple (pool.pair landing.1 landing.2)) (.ref address)) :
    Valid heap (insert entries landing address weight) := by
  induction entries with
  | nil =>
      intro entry member
      have same : entry = ⟨landing, address, weight⟩ := by simpa [insert] using member
      subst entry
      exact key
  | cons entry rest ih =>
      have tail : Valid heap rest := fun current member => valid current (by simp [member])
      intro current member
      by_cases same : entry.landing = landing
      · simp only [insert, if_pos same, List.mem_cons] at member
        rcases member with rfl | member
        · exact valid entry (by simp)
        · exact tail current member
      · simp only [insert, if_neg same, List.mem_cons] at member
        rcases member with rfl | member
        · exact valid current (by simp)
        · exact ih tail current member

end solvers.«84».resolve_landing.weights
