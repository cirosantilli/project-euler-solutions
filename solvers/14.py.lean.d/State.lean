import «.formalsw».solvers.«14.py.ast»
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.ScalarDicts
import ProjectEulerStatements.Lib.ArgMaximum
import ProjectEulerStatements.Lib.Collatz

/-! Shared cache, heap, and local-variable invariants for the optimized Collatz solver.
Source body selectors and execution proofs remain in the main solver module. -/

namespace solvers.«14»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.P14

abbrev program := Solvers.P14Ast.module

namespace Cache
open ProjectEulerStatements.Collatz

abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

/-- List entries use zero as a miss; the dictionary contains only exact lengths. -/
structure Valid (limit : Nat) (small : List Mutable.Value)
    (big : List (Constant × Mutable.Value)) : Prop where
  positive : 0 < limit
  size : small.length = limit + 1
  one : small[1]? = some (number 1)
  small_values : ∀ n value, small[n]? = some value →
    ∃ length : Nat, value = number length ∧ CachedLength n length
  big_values : ∀ key value, (key, value) ∈ big →
    ∃ n : Nat, limit < n ∧ key = .int n ∧ value = number (collatzLen n)

variable {limit : Nat} {small : List Mutable.Value} {big : List (Constant × Mutable.Value)}

theorem small_update
    (valid : ∀ n value, small[n]? = some value →
      ∃ length : Nat, value = number length ∧ CachedLength n length) (position : Nat) :
    ∀ n value, (small.set position (number (collatzLen position)))[n]? = some value →
      ∃ length : Nat, value = number length ∧ CachedLength n length := by
  intro n value found
  by_cases same : position = n
  · subst n
    simp only [List.getElem?_set, ↓reduceIte] at found
    split at found
    · cases found
      exact ⟨collatzLen position, rfl, Or.inr rfl⟩
    · contradiction
  · rw [List.getElem?_set_ne same] at found
    exact valid n value found

theorem big_lookup (valid : ∀ key value, (key, value) ∈ big →
      ∃ n : Nat, limit < n ∧ key = .int n ∧ value = number (collatzLen n))
    (n : Nat) (found : ScalarDict.lookup big (.int n) = some value) :
    value = number (collatzLen n) := by
  induction big with
  | nil => simp [ScalarDict.lookup] at found
  | cons pair rest ih =>
      rcases pair with ⟨key, stored⟩
      obtain ⟨m, _, rfl, rfl⟩ := valid key stored (by simp)
      simp only [ScalarDict.lookup, scalarCompare, evalCmpOp] at found
      by_cases same : m = n
      · subst m
        simpa using found.symm
      · simp [Int.ofNat_inj, same] at found
        exact ih (by intro key value member; exact valid key value (by simp [member])) found

theorem big_insert (valid : ∀ key value, (key, value) ∈ big →
      ∃ n : Nat, limit < n ∧ key = .int n ∧ value = number (collatzLen n))
    (n : Nat) (large : limit < n) :
    ∀ key value, (key, value) ∈ ScalarDict.insert big (.int n) (number (collatzLen n)) →
      ∃ m : Nat, limit < m ∧ key = .int m ∧ value = number (collatzLen m) := by
  induction big with
  | nil =>
      intro key value member
      simp only [ScalarDict.insert, List.mem_singleton, Prod.mk.injEq] at member
      obtain ⟨rfl, rfl⟩ := member
      exact ⟨n, large, rfl, rfl⟩
  | cons pair rest ih =>
      rcases pair with ⟨key, stored⟩
      obtain ⟨m, outside, rfl, rfl⟩ := valid key stored (by simp)
      have tail : ∀ key value, (key, value) ∈ rest →
          ∃ m : Nat, limit < m ∧ key = .int m ∧ value = number (collatzLen m) := by
        intro key value member
        exact valid key value (by simp [member])
      intro key value member
      by_cases same : m = n
      · subst m
        simp [ScalarDict.insert, scalarCompare, evalCmpOp] at member
        rcases member with ⟨rfl, rfl⟩ | member
        · exact ⟨n, large, rfl, rfl⟩
        · exact tail key value member
      · simp [ScalarDict.insert, scalarCompare, evalCmpOp, Int.ofNat_inj, same] at member
        rcases member with ⟨rfl, rfl⟩ | member
        · exact ⟨m, outside, rfl, rfl⟩
        · exact ih tail key value member

theorem Valid.update_small (valid : Valid limit small big) (n : Nat) :
    Valid limit (small.set n (number (collatzLen n))) big := by
  have positive := valid.positive
  refine ⟨valid.positive, by simpa using valid.size, ?_, small_update valid.small_values n, valid.big_values⟩
  by_cases one : n = 1
  · subst n
    simp [length_one, show 1 < small.length by rw [valid.size]; omega]
  · simpa [List.getElem?_set_ne one] using valid.one

theorem Valid.update_big (valid : Valid limit small big) (n : Nat) (large : limit < n) :
    Valid limit small (ScalarDict.insert big (.int n) (number (collatzLen n))) :=
  ⟨valid.positive, valid.size, valid.one, valid.small_values, big_insert valid.big_values n large⟩

def initialSmall (limit : Nat) : List Mutable.Value :=
  (List.replicate (limit + 1) (number 0)).set 1 (number 1)

theorem initial_valid (limit : Nat) (positive : 0 < limit) : Valid limit (initialSmall limit) [] := by
  have zeros : ∀ n value, (List.replicate (limit + 1) (number 0))[n]? = some value →
      ∃ length : Nat, value = number length ∧ CachedLength n length := by
    intro n value found
    rw [List.getElem?_replicate] at found
    split at found
    · cases found
      exact ⟨0, rfl, Or.inl rfl⟩
    · contradiction
  refine ⟨positive, by simp [initialSmall], ?_, ?_, by simp⟩
  · simp [initialSmall, show 1 < limit + 1 by omega]
  · simpa [initialSmall, length_one] using small_update zeros 1

/-- Every small-cache probe succeeds and gives either a miss or the exact length. -/
theorem Valid.small_probe (valid : Valid limit small big) (n : Nat) (inside : n ≤ limit) :
    ∃ length : Nat, small[n]? = some (number length) ∧ CachedLength n length := by
  have bound : n < small.length := by rw [valid.size]; omega
  obtain ⟨length, value, correct⟩ := valid.small_values n small[n] (List.getElem?_eq_getElem bound)
  exact ⟨length, by simpa only [value] using List.getElem?_eq_getElem bound, correct⟩

theorem Valid.big_probe (valid : Valid limit small big) (n : Nat) :
    ScalarDict.lookup big (.int n) = none ∨
    ScalarDict.lookup big (.int n) = some (number (collatzLen n)) := by
  cases found : ScalarDict.lookup big (.int n) with
  | none => exact Or.inl rfl
  | some value => exact Or.inr (congrArg some (big_lookup valid.big_values n found))

/-- The designated base entry prevents a miss at 1. -/
theorem Valid.miss_not_one (valid : Valid limit small big) (n : Nat)
    (miss : if n ≤ limit then small[n]? = some (number 0) else ScalarDict.lookup big (.int n) = none) :
    n ≠ 1 := by
  have positive := valid.positive
  intro one
  subst n
  simp [show 1 ≤ limit by omega, valid.one, number] at miss

/-- The heap write keeps the list object at its original address. -/
theorem write_small (heap : Mutable.Heap) (address n : Nat) (items : List Mutable.Value)
    (value : Mutable.Value) (found : heap[address]? = some (.list items)) (inside : n < items.length) :
    Mutable.setItem (.ref address) (number n) value heap =
      some (.ok (.scalar .none), heap.set address (.list (items.set n value))) := by
  have nonnegative : ¬ (n : Int) < 0 := by omega
  have bounded : ¬ items.length ≤ n := by omega
  simp [Mutable.setItem, Mutable.index, found, number, nonnegative, bounded, Except.map]


def Known (limit : Nat) (small : List Mutable.Value) (big : List (Constant × Mutable.Value)) (n : Nat) : Prop :=
  if n ≤ limit then small[n]? = some (number (collatzLen n))
  else ScalarDict.lookup big (.int n) = some (number (collatzLen n))

def Missing (limit : Nat) (small : List Mutable.Value) (big : List (Constant × Mutable.Value)) (n : Nat) : Prop :=
  if n ≤ limit then small[n]? = some (number 0) else ScalarDict.lookup big (.int n) = none

theorem Valid.classify (valid : Valid limit small big) (n : Nat) :
    Known limit small big n ∨ Missing limit small big n := by
  by_cases inside : n ≤ limit
  · obtain ⟨length, found, correct⟩ := valid.small_probe n inside
    rcases correct with zero | exactLength
    · exact Or.inr (by simpa [Missing, inside, zero] using found)
    · exact Or.inl (by simpa [Known, inside, exactLength] using found)
  · rcases valid.big_probe n with missing | known
    · exact Or.inr (by simpa [Missing, inside] using missing)
    · exact Or.inl (by simpa [Known, inside] using known)

end Cache

def storedHeap (limit n : Nat) (heap : Mutable.Heap) (smallAddress bigAddress : Nat)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value)) : Mutable.Heap :=
  if n ≤ limit then heap.set smallAddress (.list (small.set n (.scalar (.int (collatzLen n)))))
  else heap.set bigAddress (.dict (ScalarDict.insert big (.int n) (.scalar (.int (collatzLen n)))))

/-- A cache write preserves the other cache and establishes an exact entry for the vertex. -/
theorem storedHeap_valid (limit n : Nat) (heap : Mutable.Heap) (smallAddress bigAddress : Nat)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big) (distinct : smallAddress ≠ bigAddress)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big)) :
    ∃ updatedSmall updatedBig,
      (storedHeap limit n heap smallAddress bigAddress small big)[smallAddress]? = some (.list updatedSmall) ∧
      (storedHeap limit n heap smallAddress bigAddress small big)[bigAddress]? = some (.dict updatedBig) ∧
      Cache.Valid limit updatedSmall updatedBig ∧
      (if n ≤ limit then updatedSmall[n]? = some (.scalar (.int (collatzLen n)))
       else ScalarDict.lookup updatedBig (.int n) = some (.scalar (.int (collatzLen n)))) := by
  have smallBound := (List.getElem?_eq_some_iff.mp smallFound).1
  have bigBound := (List.getElem?_eq_some_iff.mp bigFound).1
  by_cases inside : n ≤ limit
  · refine ⟨small.set n (.scalar (.int (collatzLen n))), big, ?_, ?_, valid.update_small n, ?_⟩
    · simp [storedHeap, inside, smallBound]
    · simpa [storedHeap, inside, List.getElem?_set_ne distinct] using bigFound
    · simp [inside, show n < small.length by rw [valid.size]; omega]
  · refine ⟨small, ScalarDict.insert big (.int n) (.scalar (.int (collatzLen n))), ?_, ?_,
      valid.update_big n (by omega), ?_⟩
    · simpa [storedHeap, inside, List.getElem?_set_ne distinct.symm] using smallFound
    · simp [storedHeap, inside, bigBound]
    · simp [inside, ScalarDict.lookup_insert_self]

theorem storedHeap_other (limit n : Nat) (heap : Mutable.Heap) (smallAddress bigAddress address : Nat)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (notSmall : smallAddress ≠ address) (notBig : bigAddress ≠ address) :
    (storedHeap limit n heap smallAddress bigAddress small big)[address]? = heap[address]? := by
  unfold storedHeap
  split <;> simp [List.getElem?_set_ne, notSmall, notBig]

theorem storedHeap_length (limit n : Nat) (base : Mutable.Heap) (smallAddress bigAddress : Nat)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value)) :
    (storedHeap limit n base smallAddress bigAddress small big).length = base.length := by
  unfold storedHeap
  split <;> simp

theorem storedHeap_append (limit n : Nat) (base suffix : Mutable.Heap) (smallAddress bigAddress : Nat)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (smallBound : smallAddress < base.length) (bigBound : bigAddress < base.length) :
    storedHeap limit n (base ++ suffix) smallAddress bigAddress small big =
      storedHeap limit n base smallAddress bigAddress small big ++ suffix := by
  unfold storedHeap
  split <;> simp [List.set_append_left, smallBound, bigBound]

def missValue (limit n : Nat) : Mutable.Value := if n ≤ limit then .scalar (.int 0) else .scalar .none

/-- Forward misses record the actual vertices, in order, without a fixed step bound. -/
inductive Trail (origin : Nat) : Nat → List Nat → Prop where
  | start : Trail origin origin []
  | step {current : Nat} {path : List Nat} : Trail origin current path →
      0 < current → current ≠ 1 → Trail origin (collatzStep current) (path ++ [current])

def vertices (path : List Nat) : List Mutable.Value := path.map (fun n => .scalar (.int n))

structure WalkLocals (env : Mutable.Env) (origin limit smallAddress bigAddress pathAddress current : Nat) : Prop where
  origin : env "n" = some (.scalar (.int origin))
  bound : env "limit" = some (.scalar (.int limit))
  small : env "cache_small" = some (.ref smallAddress)
  big : env "cache_big" = some (.ref bigAddress)
  path : env "path" = some (.ref pathAddress)
  current : env "m" = some (.scalar (.int current))

theorem WalkLocals.miss {env : Mutable.Env} {origin limit smallAddress bigAddress pathAddress current : Nat}
    (locals : WalkLocals env origin limit smallAddress bigAddress pathAddress current) :
    WalkLocals (setLocal (setLocal env "cm" (missValue limit current)) "m" (.scalar (.int (collatzStep current))))
      origin limit smallAddress bigAddress pathAddress (collatzStep current) := by
  rcases locals with ⟨hn, hl, hs, hb, hp, hm⟩
  constructor <;> simp_all [setLocal]

theorem WalkLocals.hit {env : Mutable.Env} {origin limit smallAddress bigAddress pathAddress current : Nat}
    (locals : WalkLocals env origin limit smallAddress bigAddress pathAddress current) :
    WalkLocals (setLocal (setLocal env "cm" (.scalar (.int (collatzLen current)))) "length" (.scalar (.int (collatzLen current))))
      origin limit smallAddress bigAddress pathAddress current := by
  rcases locals with ⟨hn, hl, hs, hb, hp, hm⟩
  constructor <;> simp_all [setLocal]

theorem WalkLocals.unwind {env : Mutable.Env} {origin limit smallAddress bigAddress pathAddress current : Nat}
    (locals : WalkLocals env origin limit smallAddress bigAddress pathAddress current) (n : Nat) :
    WalkLocals (setLocal (setLocal env "v" (.scalar (.int n))) "length" (.scalar (.int (collatzLen n))))
      origin limit smallAddress bigAddress pathAddress current := by
  rcases locals with ⟨hn, hl, hs, hb, hp, hm⟩
  constructor <;> simp_all [setLocal]

structure SearchLocals (env : Mutable.Env) (limit smallAddress bigAddress winner : Nat) : Prop where
  bound : env "limit" = some (.scalar (.int limit))
  small : env "cache_small" = some (.ref smallAddress)
  big : env "cache_big" = some (.ref bigAddress)
  best : env "best_n" = some (.scalar (.int winner))
  length : env "best_len" = some (.scalar (.int (collatzLen winner)))

def searchHeap (limit : Nat) : Mutable.Heap :=
  [.list [.scalar (.int 0)], .list (Cache.initialSmall limit), .dict []]

theorem initial_winner (n : Nat) (positive : 0 < n) :
    (List.range' 1 n).foldl (ProjectEulerStatements.ArgMaximum.select collatzLen) 1 =
      ProjectEulerStatements.P14.naive n := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : n ≠ 0)
  simp [ProjectEulerStatements.P14.naive, ProjectEulerStatements.ArgMaximum.argmax_range_succ,
    List.range'_succ, ProjectEulerStatements.ArgMaximum.select,
    ProjectEulerStatements.Collatz.length_zero, ProjectEulerStatements.Collatz.length_one]

end solvers.«14»
