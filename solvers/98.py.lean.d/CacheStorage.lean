import solvers.«98.py.lean.d».SquareCoverage
import ProjectEulerSolutions.Lib.KeyedReferences

namespace solvers.«98».square_cache
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open pattern_signature (program)
open squares_with_length (signature squares setValues record)
set_option Elab.async false

abbrev Rows := List (Int × Nat)
abbrev work := KeyedReferences.work KeyedReferences.integers
abbrev number (n : Int) : Mutable.Value := .scalar (.int n)

/-- A cache entry retains the returned record, its membership set, and every pattern bucket. -/
def Stored (outer : Nat) (heap : Mutable.Heap) (width : Int) (address : Nat) : Prop :=
  ∃ dictionary buckets, outer < dictionary ∧ PatternBuckets.View heap dictionary buckets ∧
    PatternBuckets.Partition signature (squares width.toNat) buckets ∧
    heap[dictionary + 1]? = some (.set (setValues (squares width.toNat))) ∧
    heap[address]? = some (record dictionary (dictionary + 1))

def Valid (outer : Nat) (heap : Mutable.Heap) (rows : Rows) : Prop :=
  ∀ row ∈ rows, Stored outer heap row.1 row.2

theorem Stored.append {outer : Nat} {heap : Mutable.Heap} {width : Int} {address : Nat}
    (stored : Stored outer heap width address) (extra : Mutable.Heap) : Stored outer (heap ++ extra) width address := by
  obtain ⟨dictionary, buckets, afterOuter, view, partition, foundSet, foundRecord⟩ := stored
  exact ⟨dictionary, buckets, afterOuter, view.append extra, partition,
    HeapCollection.prefix_found heap extra _ _ foundSet, HeapCollection.prefix_found heap extra _ _ foundRecord⟩

theorem Stored.write {outer : Nat} {heap : Mutable.Heap} {width : Int} {address : Nat}
    (stored : Stored outer heap width address) (old replacement : List (Constant × Mutable.Value))
    (found : heap[outer]? = some (.dict old)) : Stored outer (heap.set outer (.dict replacement)) width address := by
  obtain ⟨dictionary, buckets, afterOuter, view, partition, foundSet, foundRecord⟩ := stored
  refine ⟨dictionary, buckets, afterOuter, ?_, partition, ?_, ?_⟩
  · refine ⟨?_, ?_, view.afterDictionary, view.unique⟩
    · rw [List.getElem?_set_ne (by omega : outer ≠ dictionary)]
      exact view.found
    · intro bucket member
      have afterDictionary := view.afterDictionary bucket member
      rw [List.getElem?_set_ne (by omega : outer ≠ bucket.address)]
      exact view.lists bucket member
  · rw [List.getElem?_set_ne (by omega : outer ≠ dictionary + 1)]
    exact foundSet
  · have different : outer ≠ address := by
      intro same
      rw [← same, found] at foundRecord
      cases foundRecord
    rw [List.getElem?_set_ne different]
    exact foundRecord

theorem Valid.append {outer : Nat} {heap : Mutable.Heap} {rows : Rows}
    (valid : Valid outer heap rows) (extra : Mutable.Heap) : Valid outer (heap ++ extra) rows :=
  fun row member => (valid row member).append extra

abbrev body (collector : Nat) : List Stmt :=
  [.comprehensionStore collector (.name "L") (.call (.name "squares_with_length") [.name "L"])]

/-- Each actual helper call can allocate freely before its record is stored in the outer dictionary. -/
theorem insert (base : Mutable.Heap) (rows : Rows) (garbage : Mutable.Heap) (width : Int) (positive : 0 < width)
    (env : HeapCalls.Env) (getWidth : env "L" = some (number width)) (valid : Valid base.length (work base rows garbage) rows) :
    ∃ extra address,
      BlockContinues program env (body base.length) (work base rows garbage) env
        (work base (KeyedReferences.insert rows width address) (garbage ++ extra)) ∧
      Valid base.length (work base (KeyedReferences.insert rows width address) (garbage ++ extra))
        (KeyedReferences.insert rows width address) := by
  have cast : (width.toNat : Int) = width := by omega
  obtain ⟨extra, buckets, address, returned, view, partition, foundSet, foundRecord⟩ :=
    squares_with_length.result width.toNat (by omega) (work base rows garbage)
  have callResult : ReturnsValue program "squares_with_length" [number width] (work base rows garbage) (.ref address)
      (work base rows garbage ++ extra) := by simpa only [pattern_signature.number, cast] using returned
  have stored : Stored base.length (work base rows garbage ++ extra) width address := by
    refine ⟨(work base rows garbage).length, buckets, ?_, view, partition, foundSet, foundRecord⟩
    simp [work, KeyedReferences.work]
  have found := KeyedReferences.found KeyedReferences.integers base rows (garbage ++ extra)
  have written : Mutable.setItem (.ref base.length) (number width) (.ref address) (work base rows garbage ++ extra) =
      some (.ok (.scalar .none), work base (KeyedReferences.insert rows width address) (garbage ++ extra)) := by
    simpa only [KeyedReferences.append] using
      KeyedReferences.set_item KeyedReferences.integers base rows (garbage ++ extra) width address
  refine ⟨extra, address, ?_, ?_⟩
  · exact BlockContinues.comprehension_store (.cons (Evaluates.lookup getWidth)
      (.cons (Evaluates.call (.cons (Evaluates.lookup getWidth) .nil) callResult) .nil)) written
  · intro row member
    have oldValid := valid.append extra
    have selected : Stored base.length (work base rows garbage ++ extra) row.1 row.2 := by
      rcases KeyedReferences.mem_insert rows width address row member with old | fresh
      · exact oldValid row old
      · simpa only [fresh] using stored
    have kept := selected.write (KeyedReferences.entries KeyedReferences.integers rows)
      (KeyedReferences.entries KeyedReferences.integers (KeyedReferences.insert rows width address))
      (by simpa only [KeyedReferences.append] using found)
    simpa only [KeyedReferences.append, KeyedReferences.store] using kept

end solvers.«98».square_cache
