import solvers.«92.py.lean.d».Digits
import lang.py.FormalSW.KeyDictionaryCalls
import ProjectEulerSolutions.Lib.DigitSquareDistribution

namespace solvers.«92».table
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev number := next_term.number
abbrev Entry := Prefix.State × Nat × Nat
abbrev Entries := List Entry

def key (state : Prefix.State) : HashKey :=
  .tuple [.scalar (.int state.1), .scalar (.bool state.2)]

def tuple (state : Prefix.State) : Mutable.Object :=
  .tuple [number state.1, .scalar (.bool state.2)]

/-- Each dictionary entry retains the address of its first inserted tuple key. -/
def encode (entries : Entries) : List (HashDict.Entry Mutable.Value) :=
  entries.map fun (state, address, ways) => (key state, .ref address, number ways)

def view (entries : Entries) : Prefix.Table :=
  entries.map fun (state, _, ways) => (state, ways)

def lookup : Entries → Prefix.State → Option Nat
  | [], _ => none
  | (stored, _, ways) :: rest, state =>
      if stored = state then some ways else lookup rest state

def insert : Entries → Prefix.State → Nat → Nat → Entries
  | [], state, address, ways => [(state, address, ways)]
  | (stored, original, count) :: rest, state, address, ways =>
      if stored = state then (stored, original, ways) :: rest
      else (stored, original, count) :: insert rest state address ways

def add (entries : Entries) (state : Prefix.State) (address ways : Nat) : Entries :=
  insert entries state address ((lookup entries state).getD 0 + ways)

theorem key_equal (left right : Prefix.State) :
    HashKey.equal (key left) (key right) = decide (left = right) := by
  rcases left with ⟨a, b⟩
  rcases right with ⟨c, d⟩
  cases b <;> cases d <;>
    simp [key, HashKey.equal, HashKey.depth, HashKey.listDepth, HashKey.equalFuel,
      List.isEqv, scalarCompare, evalCmpOp, Prod.mk.injEq]

theorem lookup_encode (entries : Entries) (state : Prefix.State) :
    HashDict.lookup (encode entries) (key state) = (lookup entries state).map number := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      rcases entry with ⟨stored, address, ways⟩
      simp only [encode, List.map_cons, HashDict.lookup_cons, key_equal, lookup]
      split <;> simp_all [encode]

theorem insert_encode (entries : Entries) (state : Prefix.State) (address ways : Nat) :
    HashDict.insert (encode entries) (key state) (.ref address) (number ways) =
      encode (insert entries state address ways) := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      rcases entry with ⟨stored, original, count⟩
      simp only [encode, List.map_cons, HashDict.insert, key_equal, insert]
      split <;> simp_all [encode]

/-- Reading the old count and assigning its increment implements the mathematical merge. -/
theorem view_add (entries : Entries) (state : Prefix.State) (address ways : Nat) :
    view (add entries state address ways) = Prefix.add (view entries) state ways := by
  induction entries with
  | nil => simp [view, add, insert, lookup, Prefix.add]
  | cons entry rest ih =>
      rcases entry with ⟨stored, original, count⟩
      by_cases same : stored = state
      · simp [add, insert, lookup, view, Prefix.add, same]
      · simpa [add, insert, lookup, view, Prefix.add, same] using
          congrArg (List.cons (stored, count)) ih

theorem weighted_add (entries : Entries) (state : Prefix.State) (address ways : Nat)
    (score : Prefix.State → Nat) :
    Prefix.weighted (view (add entries state address ways)) score =
      Prefix.weighted (view entries) score + ways * score state := by
  rw [view_add, Prefix.weighted_add]

/-- The initial empty dictionary has not yet been promoted to tuple keys. -/
def Found (heap : Mutable.Heap) (address : Nat) (entries : Entries) : Prop :=
  heap[address]? = some (.keyDict (encode entries)) ∨
    (entries = [] ∧ heap[address]? = some (.dict []))

def Keys (heap : Mutable.Heap) (entries : Entries) : Prop :=
  ∀ entry ∈ entries, heap[entry.2.1]? = some (tuple entry.1)

theorem Found.inside {heap : Mutable.Heap} {address : Nat} {entries : Entries}
    (found : Found heap address entries) : address < heap.length := by
  rcases found with found | ⟨_, found⟩ <;> exact (List.getElem?_eq_some_iff.mp found).1

theorem Found.append {heap : Mutable.Heap} {address : Nat} {entries : Entries}
    (found : Found heap address entries) (extra : Mutable.Heap) :
    Found (heap ++ extra) address entries := by
  have same := List.getElem?_append_left (l₂ := extra) found.inside
  rcases found with found | ⟨rfl, found⟩
  · exact Or.inl (same.trans found)
  · exact Or.inr ⟨rfl, same.trans found⟩

theorem Found.set {heap : Mutable.Heap} {address : Nat} {entries : Entries}
    (found : Found heap address entries) (replacement : Entries) :
    Found (heap.set address (.keyDict (encode replacement))) address replacement :=
  Or.inl (List.getElem?_set_self found.inside)

theorem Found.other {heap : Mutable.Heap} {address other : Nat} {entries : Entries}
    (found : Found heap address entries) (object : Mutable.Object) (different : other ≠ address) :
    Found (heap.set other object) address entries := by
  rcases found with found | ⟨rfl, found⟩
  · exact Or.inl ((List.getElem?_set_ne different).trans found)
  · exact Or.inr ⟨rfl, (List.getElem?_set_ne different).trans found⟩

theorem Keys.append {heap : Mutable.Heap} {entries : Entries}
    (keys : Keys heap entries) (extra : Mutable.Heap) : Keys (heap ++ extra) entries := by
  intro entry member
  have found := keys entry member
  exact (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found

theorem Keys.insert {heap : Mutable.Heap} {entries : Entries}
    (keys : Keys heap entries) (state : Prefix.State) (address ways : Nat)
    (found : heap[address]? = some (tuple state)) :
    Keys heap (insert entries state address ways) := by
  induction entries with
  | nil => simpa only [Keys, table.insert, List.mem_singleton, forall_eq] using found
  | cons entry rest ih =>
      rcases entry with ⟨stored, original, count⟩
      have first := keys (stored, original, count) (by simp)
      have tail : Keys heap rest := by intro e member; exact keys e (by simp [member])
      by_cases same : stored = state
      · simpa only [Keys, table.insert, if_pos same, List.mem_cons, forall_eq_or_imp] using And.intro first tail
      · simpa only [Keys, table.insert, if_neg same, List.mem_cons, forall_eq_or_imp] using And.intro first (ih tail)

theorem Keys.set {heap : Mutable.Heap} {entries original : Entries} {address : Nat}
    (keys : Keys heap entries) (found : Found heap address original) (replacement : Entries) :
    Keys (heap.set address (.keyDict (encode replacement))) entries := by
  intro entry member
  have keyFound := keys entry member
  have different : address ≠ entry.2.1 := by
    intro same
    rcases found with found | ⟨_, found⟩ <;>
      rw [same, keyFound] at found <;> simp [tuple] at found
  exact (List.getElem?_set_ne different).trans keyFound

theorem hash (heap : Mutable.Heap) (address : Nat) (state : Prefix.State)
    (found : heap[address]? = some (tuple state)) :
    Mutable.hashKey (.ref address) heap = some (.ok (key state), heap) := by
  exact Mutable.hashKey_tuple_scalars heap address [.int state.1, .bool state.2] found

theorem get (heap : Mutable.Heap) (address keyAddress : Nat) (entries : Entries)
    (state : Prefix.State) (found : Found heap address entries)
    (keyFound : heap[keyAddress]? = some (tuple state)) :
    Mutable.dictionaryGet address (.ref keyAddress) (number 0) heap =
      some (.ok (number ((lookup entries state).getD 0)), heap) := by
  have hashed := hash heap keyAddress state keyFound
  rcases found with found | ⟨rfl, found⟩
  · rw [Mutable.dictionaryGet_keyed heap address (encode entries) found _ _ hashed,
      lookup_encode]
    cases lookup entries state <;> rfl
  · simp [Mutable.dictionaryGet, found, hashed, Mutable.promoteDictionary,
      lookup, Except.map]

theorem set (heap : Mutable.Heap) (address keyAddress ways : Nat) (entries : Entries)
    (state : Prefix.State) (found : Found heap address entries)
    (keyFound : heap[keyAddress]? = some (tuple state)) :
    Mutable.setItem (.ref address) (.ref keyAddress) (number ways) heap =
      some (.ok (.scalar .none), heap.set address (.keyDict (encode (insert entries state keyAddress ways)))) := by
  have hashed := hash heap keyAddress state keyFound
  rcases found with found | ⟨rfl, found⟩
  · simp [Mutable.setItem, found, hashed, insert_encode, Except.map]
  · simp [Mutable.setItem, found, hashed, Mutable.promoteDictionary,
      HashDict.insert, encode, insert, Except.map]

end solvers.«92».table
