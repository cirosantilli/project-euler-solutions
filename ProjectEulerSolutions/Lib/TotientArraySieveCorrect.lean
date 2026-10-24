import ProjectEulerSolutions.Lib.TotientArraySieve

set_option linter.style.header false

/-! The indexed array implementation satisfies the complete mathematical sieve invariant. -/
namespace ProjectEulerSolutions.TotientArraySieve
open ProjectEulerStatements.TotientLinearSieve
set_option Elab.async false

theorem encode_congr (size : Nat) (left right : Table)
    (same : ∀ n, n < size → left n = right n) : encode size left = encode size right := by
  apply Array.ext
  · simp
  · intro n leftBound rightBound
    simp only [encode, Array.getElem_ofFn]
    exact same n (by simpa using rightBound)

/-- The indexed cache traversal implements exactly the mathematical prime-product pass. -/
theorem passArray_encode (limit i : Nat) (cache : Array Nat) (position : Nat)
    (remaining : List Nat) (table : Table) (suffix : cache.toList.drop position = remaining)
    (within : i ≤ limit) :
    passArray limit i cache position (encode (limit + 1) table) =
      encode (limit + 1) (pass limit i remaining table) := by
  induction remaining generalizing position table with
  | nil =>
      have missing : cache.toList[position]? = none := by
        simpa only [suffix, List.head?_nil] using
          (List.head?_drop (l := cache.toList) (i := position)).symm
      have exhausted : cache.size ≤ position := by
        simpa using List.getElem?_eq_none_iff.mp missing
      rw [passArray, dif_neg (by omega), pass]
  | cons p rest ih =>
      have item : cache[position]? = some p := by
        simpa only [suffix, List.head?_cons, Array.getElem?_toList] using
          (List.head?_drop (l := cache.toList) (i := position)).symm
      obtain ⟨bounded, value⟩ := Array.getElem?_eq_some_iff.mp item
      rw [passArray, dif_pos bounded, value, pass]
      by_cases outside : limit < i * p
      · simp only [if_pos outside]
      · simp only [if_neg outside, writeCell_encode limit i p table within]
        by_cases divides : i % p = 0
        · simp only [if_pos divides]
        · simp only [if_neg divides]
          exact ih (position + 1) (write i p table)
            (by simp [List.drop_add_one_eq_tail_drop, suffix])

theorem prepareArray_encode (size i : Nat) (large : 2 ≤ i) (within : i < size)
    (cache : Array Nat) :
    prepareArray i (encode size (expected i)) cache =
      (encode size (prepare i (expected i)), if Nat.Prime i then cache.push i else cache) := by
  by_cases prime : Nat.Prime i
  · simp only [prepareArray, encode_read size i (expected i) within, prepare,
      expected_current i large, if_pos prime, Bool.false_eq_true, if_false]
    rw [encode_set]
  · simp only [prepareArray, encode_read size i (expected i) within, prepare,
      expected_current i large, if_neg prime, if_true]

theorem initialCells_encode (limit : Nat) :
    initialCells limit = encode (limit + 1) (expected 2) := by
  apply Array.ext
  · simp [initialCells]
  · intro n leftBound rightBound
    have bounded : n < limit + 1 := by simpa using rightBound
    simp only [initialCells, Array.set!_eq_setIfInBounds]
    rw [Array.getElem_setIfInBounds (by simpa using bounded)]
    simp [encode, expected_initial, eq_comm]

/-- Every outer iteration advances the complete table and ordered prime cache. -/
theorem runArray_correct (limit i : Nat) (cache : Array Nat) (large : 2 ≤ i)
    (within : i ≤ limit + 1) (ordered : cache.toList.Pairwise (· < ·))
    (complete : ∀ p, p ∈ cache.toList ↔ Nat.Prime p ∧ p < i) :
    runArray limit i (encode (limit + 1) (expected i)) cache =
      encode (limit + 1) (expected (limit + 1)) := by
  by_cases active : i ≤ limit
  · let advanced : Array Nat := if Nat.Prime i then cache.push i else cache
    have listEq : advanced.toList = nextCache i cache.toList := by
      by_cases prime : Nat.Prime i <;> simp [advanced, nextCache, prime]
    have orderedNext : advanced.toList.Pairwise (· < ·) := by
      rw [listEq]
      exact nextCache_ordered i cache.toList ordered complete
    have completeNext : ∀ p, p ∈ advanced.toList ↔ Nat.Prime p ∧ p < i + 1 := by
      simpa only [listEq, Nat.lt_succ_iff] using nextCache_complete i cache.toList complete
    have mathematical := step_correct limit i advanced.toList (expected i) large active
      orderedNext (by simpa only [Nat.lt_succ_iff] using completeNext) (fun _ _ => rfl)
    have tableEq := encode_congr (limit + 1)
      (pass limit i advanced.toList (prepare i (expected i))) (expected (i + 1))
      (fun n bounded => mathematical n (by omega))
    rw [runArray, if_pos active, prepareArray_encode (limit + 1) i large (by omega)]
    change runArray limit (i + 1)
      (passArray limit i advanced 0 (encode (limit + 1) (prepare i (expected i)))) advanced = _
    rw [passArray_encode limit i advanced 0 advanced.toList _ rfl active, tableEq]
    exact runArray_correct limit (i + 1) advanced (by omega) (by omega) orderedNext completeNext
  · have finished : i = limit + 1 := by omega
    rw [runArray, if_neg active, finished]
termination_by limit + 1 - i
decreasing_by omega

theorem computeCells_encode (limit : Nat) (positive : 1 ≤ limit) :
    computeCells limit = encode (limit + 1) (expected (limit + 1)) := by
  rw [computeCells, initialCells_encode]
  apply runArray_correct limit 2 #[] (by decide) (by omega) (by simp)
  intro p
  simp only [List.not_mem_nil, false_iff, not_and]
  intro prime
  have lower := prime.two_le
  omega

/-- Every stored array value is the mathematical totient at that index. -/
theorem computePhiLinear_correct (limit n : Nat) (bounded : n ≤ limit) :
    (computePhiLinear limit)[n]! = n.totient := by
  by_cases positive : 1 ≤ limit
  · rw [computePhiLinear, computeCells_encode limit positive]
    have within : n < (encode (limit + 1) (expected (limit + 1))).size := by simp; omega
    rw [getElem!_pos _ _ (by simpa using within), Array.getElem_map]
    simpa only [encode, Array.getElem_ofFn] using expected_finished limit n bounded
  · have zeroLimit : limit = 0 := by omega
    have zeroN : n = 0 := by omega
    subst limit n
    simp [computePhiLinear, computeCells, runArray, initialCells]

theorem computePhiLinear_size (limit : Nat) : (computePhiLinear limit).size = limit + 1 := by
  by_cases positive : 1 ≤ limit
  · simp [computePhiLinear, computeCells_encode limit positive]
  · have zero : limit = 0 := by omega
    subst limit
    simp [computePhiLinear, computeCells, runArray, initialCells]

end ProjectEulerSolutions.TotientArraySieve
