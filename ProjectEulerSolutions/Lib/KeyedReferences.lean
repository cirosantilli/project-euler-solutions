import lang.py.FormalSW.ComprehensionCalls
import ProjectEulerSolutions.Lib.HeapCollection

/-! Scalar-key dictionaries that retain shared references to separately allocated objects. -/
namespace ProjectEulerSolutions.KeyedReferences
open FormalSW.PyAst HeapCalls
set_option Elab.async false

structure Encoding (α : Type) [DecidableEq α] where
  scalar : α → Constant
  compares : ∀ a b, scalarCompare .eq (scalar a) (scalar b) = some (decide (a = b))

def integers : Encoding Int := ⟨Constant.int, by intros; simp [scalarCompare, evalCmpOp]⟩
def strings : Encoding String := ⟨Constant.string, by intros; simp [scalarCompare]⟩

variable {α : Type} [DecidableEq α]

def lookup : List (α × Nat) → α → Option Nat
  | [], _ => none
  | (key, address) :: rest, query => if key = query then some address else lookup rest query

def insert : List (α × Nat) → α → Nat → List (α × Nat)
  | [], key, address => [(key, address)]
  | (old, previous) :: rest, key, address =>
      if old = key then (old, address) :: rest else (old, previous) :: insert rest key address

def entries (encoding : Encoding α) (rows : List (α × Nat)) : List (Constant × Mutable.Value) :=
  rows.map fun row => (encoding.scalar row.1, .ref row.2)

theorem lookup_entries (encoding : Encoding α) (rows : List (α × Nat)) (key : α) :
    ScalarDict.lookup (entries encoding rows) (encoding.scalar key) = (lookup rows key).map Mutable.Value.ref := by
  induction rows with
  | nil => rfl
  | cons row rest ih =>
      rcases row with ⟨old, address⟩
      by_cases same : old = key <;> simp_all [entries, ScalarDict.lookup, encoding.compares, lookup]

theorem insert_entries (encoding : Encoding α) (rows : List (α × Nat)) (key : α) (address : Nat) :
    ScalarDict.insert (entries encoding rows) (encoding.scalar key) (.ref address) = entries encoding (insert rows key address) := by
  induction rows with
  | nil => rfl
  | cons row rest ih =>
      rcases row with ⟨old, previous⟩
      by_cases same : old = key <;> simp_all [entries, ScalarDict.insert, encoding.compares, insert]

theorem lookup_insert (rows : List (α × Nat)) (key query : α) (address : Nat) :
    lookup (insert rows key address) query = if query = key then some address else lookup rows query := by
  induction rows with
  | nil => by_cases same : key = query <;> simp [lookup, insert, same, Ne.symm]
  | cons row rest ih =>
      rcases row with ⟨old, previous⟩
      by_cases same : old = key <;> by_cases queried : old = query <;> by_cases chosen : query = key <;>
        simp_all [lookup, insert]

theorem mem_insert (rows : List (α × Nat)) (key : α) (address : Nat) (row : α × Nat)
    (member : row ∈ insert rows key address) : row ∈ rows ∨ row = (key, address) := by
  induction rows with
  | nil => exact Or.inr (by simpa [insert] using member)
  | cons first rest ih =>
      rcases first with ⟨old, previous⟩
      by_cases same : old = key
      · subst old
        have selected : row = (key, address) ∨ row ∈ rest := by simpa [insert] using member
        rcases selected with equal | member
        · exact Or.inr equal
        · exact Or.inl (List.mem_cons_of_mem _ member)
      · simp only [insert, if_neg same, List.mem_cons] at member
        rcases member with rfl | member
        · exact Or.inl List.mem_cons_self
        · exact (ih member).imp_left (List.mem_cons_of_mem _)

theorem lookup_mem (rows : List (α × Nat)) (key : α) (address : Nat) (found : lookup rows key = some address) :
    (key, address) ∈ rows := by
  induction rows with
  | nil => simp [lookup] at found
  | cons row rest ih =>
      rcases row with ⟨old, previous⟩
      by_cases same : old = key
      · subst old
        have equal : previous = address := by simpa [lookup] using found
        simp [equal]
      · exact List.mem_cons_of_mem _ (ih (by simpa only [lookup, if_neg same] using found))

def work (encoding : Encoding α) (base : Mutable.Heap) (rows : List (α × Nat)) (garbage : Mutable.Heap) : Mutable.Heap :=
  base ++ (.dict (entries encoding rows) :: garbage)

@[simp] theorem found (encoding : Encoding α) (base : Mutable.Heap) (rows : List (α × Nat)) (garbage : Mutable.Heap) :
    (work encoding base rows garbage)[base.length]? = some (.dict (entries encoding rows)) := by simp [work]

theorem append (encoding : Encoding α) (base : Mutable.Heap) (rows : List (α × Nat)) (garbage extra : Mutable.Heap) :
    work encoding base rows garbage ++ extra = work encoding base rows (garbage ++ extra) := by simp [work, List.append_assoc]

theorem store (encoding : Encoding α) (base : Mutable.Heap) (rows replacement : List (α × Nat)) (garbage : Mutable.Heap) :
    (work encoding base rows garbage).set base.length (.dict (entries encoding replacement)) = work encoding base replacement garbage := by
  simp [work, List.set_append_right]

theorem set_item (encoding : Encoding α) (base : Mutable.Heap) (rows : List (α × Nat)) (garbage : Mutable.Heap)
    (key : α) (address : Nat) :
    Mutable.setItem (.ref base.length) (.scalar (encoding.scalar key)) (.ref address) (work encoding base rows garbage) =
      some (.ok (.scalar .none), work encoding base (insert rows key address) garbage) := by
  simp [Mutable.setItem, insert_entries, store, Except.map]

/-- Updating the private dictionary preserves every other stored object and its aliases. -/
theorem other_found (encoding : Encoding α) (base : Mutable.Heap) (rows replacement : List (α × Nat)) (garbage : Mutable.Heap)
    (address : Nat) (object : Mutable.Object) (different : base.length ≠ address)
    (original : (work encoding base rows garbage)[address]? = some object) :
    (work encoding base replacement garbage)[address]? = some object := by
  rw [← store encoding base rows replacement garbage, List.getElem?_set_ne different]
  exact original

end ProjectEulerSolutions.KeyedReferences
