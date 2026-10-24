import lang.py.FormalSW.HeapSort

/-! Stable sorting of shared integer tuples, used for poker frequency groups. -/
namespace ProjectEulerSolutions.TupleSorting
open FormalSW.PyAst

set_option Elab.async false

/-- Projection is used only after proving that the referenced object is an integer tuple. -/
def readInts (heap : Mutable.Heap) : Mutable.Value → List Int
  | .ref address => match heap[address]? with
      | some (.tuple items) => items.filterMap fun value => match value with
          | .scalar (.int n) => some n
          | _ => none
      | _ => []
  | _ => []

def less (heap : Mutable.Heap) (left right : Mutable.Value) : Bool :=
  HeapSort.lexInt (readInts heap left) (readInts heap right)

def sorted (heap : Mutable.Heap) (reverse : Bool) (items : List Mutable.Value) : List Mutable.Value :=
  if reverse then (HeapSort.sortBy (less heap) items.reverse).reverse
  else HeapSort.sortBy (less heap) items

def sortedRows (reverse : Bool) (rows : List (List Int)) : List (List Int) :=
  if reverse then (HeapSort.sortBy HeapSort.lexInt rows.reverse).reverse
  else HeapSort.sortBy HeapSort.lexInt rows

/-- Reading the sorted references gives the same numerical order as sorting the rows. -/
theorem sorted_readInts (heap : Mutable.Heap) (reverse : Bool) (items : List Mutable.Value) :
    (sorted heap reverse items).map (readInts heap) = sortedRows reverse (items.map (readInts heap)) := by
  have mapped (values : List Mutable.Value) :
      (HeapSort.sortBy (less heap) values).map (readInts heap) =
        HeapSort.sortBy HeapSort.lexInt (values.map (readInts heap)) :=
    HeapSort.sortBy_map (readInts heap) HeapSort.lexInt values
  cases reverse with
  | false => exact mapped items
  | true =>
      simpa only [sorted, sortedRows, ↓reduceIte, List.map_reverse] using
        congrArg List.reverse (mapped items.reverse)

/-- Sorting rearranges the original references without copying or discarding a tuple. -/
theorem sorted_perm (heap : Mutable.Heap) (reverse : Bool) (items : List Mutable.Value) :
    (sorted heap reverse items).Perm items := by
  cases reverse with
  | false => exact HeapSort.sortBy_perm (less heap) items
  | true =>
      exact (List.reverse_perm _).trans
        ((HeapSort.sortBy_perm (less heap) items.reverse).trans (List.reverse_perm items))

theorem readInts_append (heap extra : Mutable.Heap) (address : Nat) (items : List Mutable.Value)
    (found : heap[address]? = some (.tuple items)) :
    readInts (heap ++ extra) (.ref address) = readInts heap (.ref address) := by
  have inside := (List.getElem?_eq_some_iff.mp found).1
  simp [readInts, List.getElem?_append_left inside, found]

/-- A frequency pair orders by its count and then its rank. -/
theorem less_pair (heap : Mutable.Heap) (a b : Nat) (count value otherCount otherValue : Int)
    (left : heap[a]? = some (.tuple [.scalar (.int count), .scalar (.int value)]))
    (right : heap[b]? = some (.tuple [.scalar (.int otherCount), .scalar (.int otherValue)])) :
    less heap (.ref a) (.ref b) =
      (if count = otherCount then decide (value < otherValue) else decide (count < otherCount)) := by
  by_cases same : value = otherValue
  · subst otherValue
    simp [less, readInts, left, right, HeapSort.lexInt]
  · simp [less, readInts, left, right, HeapSort.lexInt, same]

private theorem compared (heap : Mutable.Heap) (a b : Nat) (xs ys : List Int)
    (left : heap[a]? = some (.tuple (xs.map fun n => .scalar (.int n))))
    (right : heap[b]? = some (.tuple (ys.map fun n => .scalar (.int n)))) :
    HeapSort.compare (heap.length+1) heap false (.ref a) (.ref b) = some (.ok (less heap (.ref a) (.ref b))) := by
  have positive : 0 < heap.length := by
    by_cases positive : 0 < heap.length
    · exact positive
    · rw [List.getElem?_eq_none (show heap.length ≤ a by omega)] at left
      cases left
  have depth : heap.length-1+2 = heap.length+1 := by omega
  simpa [depth, less, readInts, left, right, List.filterMap_map, Function.comp_def] using
    HeapSort.compare_int_tuples heap a b (heap.length-1) xs ys left right

/-- Every typed tuple comparison succeeds, retaining the original tuple references. -/
theorem sortWithReverse (heap : Mutable.Heap) (reverse : Bool) (items : List Mutable.Value)
    (typed : ∀ item ∈ items, ∃ address : Nat, ∃ xs : List Int,
      item = .ref address ∧ heap[address]? = some (.tuple (xs.map fun n => .scalar (.int n)))) :
    HeapSort.sortWithReverse heap reverse items = some (.ok (sorted heap reverse items)) := by
  have ordered (sequence : List Mutable.Value) (included : ∀ item ∈ sequence, item ∈ items) :
      HeapSort.sort heap sequence = some (.ok (HeapSort.sortBy (less heap) sequence)) := by
    apply HeapSort.sort_of_compare
    intro left leftMember right rightMember
    obtain ⟨a, xs, rfl, leftFound⟩ := typed left (included left leftMember)
    obtain ⟨b, ys, rfl, rightFound⟩ := typed right (included right rightMember)
    exact compared heap a b xs ys leftFound rightFound
  cases reverse with
  | false => exact ordered items (fun _ member => member)
  | true =>
      have backwards := ordered items.reverse (by intro item member; simpa using member)
      simp [HeapSort.sortWithReverse, sorted, backwards, Except.map]

end ProjectEulerSolutions.TupleSorting
