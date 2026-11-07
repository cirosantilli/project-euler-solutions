import solvers.«14.py.lean.d».State
import lang.py.FormalSW.HeapMapWork

/-! Numeric bounds for the live cache and recorded Collatz trail. -/

namespace solvers.«14»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.P14 ProjectEulerStatements.Collatz

namespace Cache

/-- Every cached key and every integer query occupy one word each. -/
theorem Valid.keyWork_word_le {limit : Nat} {small : List Mutable.Value}
    {big : List (Constant × Mutable.Value)} (valid : Valid limit small big) (n : Int) :
    MapWork.keyWork .word big (.int n) ≤ 2 := by
  have foldBound (items : List Nat) (initial : Nat)
      (bounded : ∀ item ∈ items, item ≤ 1) (initialBound : initial ≤ 1) :
      items.foldl max initial ≤ 1 := by
    induction items generalizing initial with
    | nil => exact initialBound
    | cons head tail ih =>
        have headBound := bounded head (by simp)
        have tailBound : ∀ item ∈ tail, item ≤ 1 := by
          intro item member
          exact bounded item (by simp [member])
        exact ih (max initial head) tailBound (by omega)
  have widths : ∀ width ∈ big.map (fun entry => PrimitiveWork.scalarSize .word entry.1),
      width ≤ 1 := by
    intro width member
    obtain ⟨⟨key, value⟩, entryMember, rfl⟩ := List.mem_map.mp member
    obtain ⟨m, _, rfl, _⟩ := valid.big_values key value entryMember
    simp [PrimitiveWork.scalarSize, PrimitiveWork.integer]
  have widest := foldBound (big.map fun entry => PrimitiveWork.scalarSize .word entry.1)
    0 widths (by omega)
  change 1 + (big.map fun entry => PrimitiveWork.scalarSize .word entry.1).foldl max 0 ≤ 2
  omega

/-- Replacing an entry preserves the size; a missing key adds one entry. -/
theorem insert_length_le {α : Type*} (entries : List (Constant × α))
    (key : Constant) (value : α) :
    (ScalarDict.insert entries key value).length ≤ entries.length + 1 := by
  induction entries with
  | nil => simp [ScalarDict.insert]
  | cons entry rest ih =>
      obtain ⟨stored, previous⟩ := entry
      simp only [ScalarDict.insert]
      split
      · simp
      · simpa only [List.length_cons] using Nat.succ_le_succ ih

end Cache

/-- Each recorded vertex accounts for exactly one step of the original chain. -/
theorem Trail.length_add_collatzLen {origin current : Nat} {path : List Nat}
    (trail : Trail origin current path) :
    path.length + collatzLen current = collatzLen origin := by
  induction trail with
  | start => simp
  | @step current path trail positive notOne ih =>
      have next := length_step current positive notOne
      simp only [List.length_append, List.length_singleton]
      omega

theorem Trail.length_le {origin current : Nat} {path : List Nat}
    (trail : Trail origin current path) : path.length ≤ collatzLen origin := by
  have exactLength := trail.length_add_collatzLen
  omega

/-- The actual mutable path list has the same size as the recorded trail. -/
theorem Trail.vertices_length_le {origin current : Nat} {path : List Nat}
    (trail : Trail origin current path) : (solvers.«14».vertices path).length ≤ collatzLen origin := by
  simpa [solvers.«14».vertices] using trail.length_le

/-- A cache write leaves a dictionary at its address with at most one extra entry. -/
theorem storedHeap_big_length_le (limit n : Nat) (heap : Mutable.Heap)
    (smallAddress bigAddress : Nat) (small : List Mutable.Value)
    (big : List (Constant × Mutable.Value)) (distinct : smallAddress ≠ bigAddress)
    (bigFound : heap[bigAddress]? = some (.dict big)) :
    ∃ updatedBig,
      (storedHeap limit n heap smallAddress bigAddress small big)[bigAddress]? =
        some (.dict updatedBig) ∧
      updatedBig.length ≤ big.length + 1 := by
  by_cases inside : n ≤ limit
  · refine ⟨big, ?_, by omega⟩
    simpa [storedHeap, inside, List.getElem?_set_ne distinct] using bigFound
  · have bigBound := (List.getElem?_eq_some_iff.mp bigFound).1
    refine ⟨ScalarDict.insert big (.int n) (.scalar (.int (collatzLen n))), ?_,
      Cache.insert_length_le big (.int n) (.scalar (.int (collatzLen n)))⟩
    simp [storedHeap, inside, bigBound]

end solvers.«14»
