import ProjectEulerSolutions.Lib.RectangleApproximation

/-! Terminating rectangle searches with strict improvement and lexicographic ties. -/
namespace ProjectEulerSolutions.RectangleSearch

open RectangleApproximation
set_option Elab.async false

abbrev Grid := Nat × Nat

def Valid (grid : Grid) : Prop := 0 < grid.1 ∧ grid.1 ≤ grid.2
def error (target : Int) (grid : Grid) : Int := distance target grid.1 grid.2
def LexLE (left right : Grid) : Prop :=
  left.1 < right.1 ∨ left.1 = right.1 ∧ left.2 ≤ right.2
def NoWorse (target : Int) (left right : Grid) : Prop :=
  error target left < error target right ∨
    error target left = error target right ∧ LexLE left right

theorem lex_refl (grid : Grid) : LexLE grid grid := by simp [LexLE]
theorem lex_trans {a b c : Grid} (ab : LexLE a b) (bc : LexLE b c) : LexLE a c := by
  unfold LexLE at *
  omega

theorem noWorse_refl (target : Int) (grid : Grid) : NoWorse target grid grid :=
  Or.inr ⟨rfl, lex_refl grid⟩

theorem noWorse_error {target : Int} {a b : Grid} (h : NoWorse target a b) :
    error target a ≤ error target b := by
  unfold NoWorse at h
  omega

theorem noWorse_trans {target : Int} {a b c : Grid}
    (ab : NoWorse target a b) (bc : NoWorse target b c) : NoWorse target a c := by
  rcases ab with ab | ⟨equalAB, lexAB⟩
  · exact Or.inl (lt_of_lt_of_le ab (noWorse_error bc))
  · rcases bc with bc | ⟨equalBC, lexBC⟩
    · exact Or.inl (by omega)
    · exact Or.inr ⟨equalAB.trans equalBC, lex_trans lexAB lexBC⟩

theorem noWorse_antisymm {target : Int} {a b : Grid}
    (ab : NoWorse target a b) (ba : NoWorse target b a) : a = b := by
  have ea := noWorse_error ab
  have eb := noWorse_error ba
  have la : LexLE a b := by rcases ab with h | h; omega; exact h.2
  have lb : LexLE b a := by rcases ba with h | h; omega; exact h.2
  unfold LexLE at *
  exact Prod.ext (by omega) (by omega)

def choose (target : Int) (best candidate : Grid) : Grid :=
  if error target candidate < error target best then candidate else best

theorem choose_cases (target : Int) (best candidate : Grid) :
    choose target best candidate = best ∨ choose target best candidate = candidate := by
  unfold choose
  split
  · exact Or.inr rfl
  · exact Or.inl rfl

theorem choose_left (target : Int) (best candidate : Grid) :
    NoWorse target (choose target best candidate) best := by
  unfold choose
  split
  · exact Or.inl ‹_›
  · exact noWorse_refl target best

theorem choose_error (target : Int) (best candidate : Grid) :
    error target (choose target best candidate) ≤ error target best :=
  noWorse_error (choose_left target best candidate)

theorem choose_right (target : Int) {best candidate : Grid} (ordered : LexLE best candidate) :
    NoWorse target (choose target best candidate) candidate := by
  unfold choose
  split
  · exact noWorse_refl target candidate
  · by_cases equal : error target best = error target candidate
    · exact Or.inr ⟨equal, ordered⟩
    · exact Or.inl (by omega)

def potential (target : Int) (index : Nat) (best : Grid) : Nat :=
  (target + error target best).toNat + 1 - index

theorem potential_decreases (target : Int) {m n : Nat} (positive : 0 < m)
    {best next : Grid} (active : (count m n : Int) ≤ target + error target best)
    (improved : error target next ≤ error target best) :
    potential target (n + 1) next < potential target n best := by
  have lower := column_le_count positive n
  unfold potential
  omega

/-- A complete execution of the reference's dynamically bounded column loop. -/
inductive RowRun (target : Int) (m : Nat) : Nat → Grid → Grid → Prop
  | stop (n : Nat) (best : Grid) (finished : target + error target best < (count m n : Int)) :
      RowRun target m n best best
  | step (n : Nat) (best final : Grid) (active : (count m n : Int) ≤ target + error target best)
      (tail : RowRun target m (n + 1) (choose target best (m, n)) final) :
      RowRun target m n best final

theorem RowRun.improves {target : Int} {m n : Nat} {best final : Grid}
    (run : RowRun target m n best final) : NoWorse target final best := by
  induction run with
  | stop => exact noWorse_refl _ _
  | step n best final active tail ih =>
      exact noWorse_trans ih (choose_left target best (m, n))

theorem row_exists (target : Int) (m : Nat) (positive : 0 < m) (n : Nat) (best : Grid) :
    ∃ final, RowRun target m n best final := by
  by_cases active : (count m n : Int) ≤ target + error target best
  · obtain ⟨final, tail⟩ := row_exists target m positive (n + 1) (choose target best (m, n))
    exact ⟨final, .step n best final active tail⟩
  · exact ⟨best, .stop n best (by omega)⟩
termination_by potential target n best
decreasing_by exact potential_decreases target positive active (choose_error target best (m, n))

theorem RowRun.origin {target : Int} {m n : Nat} {best final : Grid}
    (run : RowRun target m n best final) : final = best ∨ final.1 = m ∧ n ≤ final.2 := by
  induction run with
  | stop => exact Or.inl rfl
  | step n best final active tail ih =>
      rcases ih with equal | ⟨row, column⟩
      · rw [equal]
        unfold choose
        split
        · exact Or.inr ⟨rfl, le_rfl⟩
        · exact Or.inl rfl
      · exact Or.inr ⟨row, by omega⟩

theorem RowRun.valid {target : Int} {m n : Nat} {best final : Grid}
    (run : RowRun target m n best final) (positive : 0 < m) (eligible : m ≤ n)
    (initial : Valid best) : Valid final := by
  rcases run.origin with equal | ⟨row, column⟩
  · simpa only [equal] using initial
  · unfold Valid
    omega

theorem RowRun.minimizes {target : Int} {m n : Nat} {best final : Grid}
    (run : RowRun target m n best final) :
    0 < m → LexLE best (m, n) → ∀ k, n ≤ k → NoWorse target final (m, k) := by
  induction run with
  | stop n best finished =>
      intro positive ordered k later
      have monotone := (count_strictMono positive).monotone later
      have absBound := le_abs_self ((count m k : Int) - target)
      apply Or.inl
      change error target best < |(count m k : Int) - target|
      omega
  | step n best final active tail ih =>
      intro positive ordered k later
      have nextOrdered : LexLE (choose target best (m, n)) (m, n + 1) := by
        unfold choose
        split <;> unfold LexLE at * <;> omega
      by_cases equal : k = n
      · subst k
        exact noWorse_trans tail.improves (choose_right target ordered)
      · exact ih positive nextOrdered k (by omega)

def fastRow (target : Int) (m : Nat) (best : Grid) : Grid :=
  let below := rowBelow target m
  choose target (choose target best (m, below)) (m, below + 1)

theorem fastRow_improves (target : Int) (m : Nat) (best : Grid) :
    NoWorse target (fastRow target m best) best :=
  noWorse_trans (choose_left target _ _) (choose_left target _ _)

theorem fastRow_origin (target : Int) (m : Nat) (best : Grid) :
    fastRow target m best = best ∨
      (fastRow target m best).1 = m ∧ m ≤ (fastRow target m best).2 := by
  have eligible : m ≤ rowBelow target m := Nat.le_max_left _ _
  rcases choose_cases target (choose target best (m, rowBelow target m))
    (m, rowBelow target m + 1) with kept | updated
  · change fastRow target m best = choose target best (m, rowBelow target m) at kept
    rcases choose_cases target best (m, rowBelow target m) with original | lower
    · exact Or.inl (kept.trans original)
    · rw [kept, lower]
      exact Or.inr ⟨rfl, eligible⟩
  · change fastRow target m best = (m, rowBelow target m + 1) at updated
    rw [updated]
    exact Or.inr ⟨rfl, by omega⟩

theorem fastRow_valid (target : Int) {m : Nat} (positive : 0 < m)
    {best : Grid} (initial : Valid best) : Valid (fastRow target m best) := by
  rcases fastRow_origin target m best with equal | ⟨row, column⟩
  · simpa only [equal] using initial
  · unfold Valid
    omega

theorem fastRow_minimizes (target : Int) {m : Nat} (positive : 0 < m)
    {best : Grid} (ordered : LexLE best (m, m)) (n : Nat) (eligible : m ≤ n) :
    NoWorse target (fastRow target m best) (m, n) := by
  have belowEligible : m ≤ rowBelow target m := Nat.le_max_left _ _
  have beforeBelow : LexLE best (m, rowBelow target m) := by unfold LexLE at *; omega
  have beforeAbove : LexLE (choose target best (m, rowBelow target m)) (m, rowBelow target m + 1) := by
    unfold choose
    split <;> unfold LexLE at * <;> omega
  have lower : NoWorse target (fastRow target m best) (m, rowBelow target m) :=
    noWorse_trans (choose_left target _ _) (choose_right target beforeBelow)
  have upper : NoWorse target (fastRow target m best) (m, rowBelow target m + 1) :=
    choose_right target beforeAbove
  rcases row_candidates_complete target positive eligible with equal | equal | less | less
  · simpa only [equal] using lower
  · simpa only [equal] using upper
  · exact Or.inl (lt_of_le_of_lt (noWorse_error lower) less)
  · exact Or.inl (lt_of_le_of_lt (noWorse_error upper) less)

/-- The two updates equal the entire dynamically bounded reference row scan. -/
theorem row_algorithms_agree {target : Int} {m : Nat} (positive : 0 < m)
    {best final : Grid} (ordered : LexLE best (m, m))
    (run : RowRun target m m best final) : final = fastRow target m best := by
  apply noWorse_antisymm
  · rcases fastRow_origin target m best with equal | ⟨row, column⟩
    · simpa only [equal] using run.improves
    · have same : fastRow target m best = (m, (fastRow target m best).2) := Prod.ext row rfl
      rw [same]
      exact run.minimizes positive ordered _ column
  · rcases run.origin with equal | ⟨row, column⟩
    · simpa only [equal] using fastRow_improves target m best
    · have same : final = (m, final.2) := Prod.ext row rfl
      rw [same]
      exact fastRow_minimizes target positive ordered _ column

/-- Properties needed by the common outer loop, independent of how a row is searched. -/
structure RowContract (target : Int) (step : Nat → Grid → Grid → Prop) : Prop where
  terminates (m : Nat) (positive : 0 < m) (best : Grid) : ∃ next, step m best next
  improves {m : Nat} {best next : Grid} : step m best next → NoWorse target next best
  origin {m : Nat} {best next : Grid} : step m best next → next = best ∨ next.1 = m ∧ m ≤ next.2
  minimizes {m : Nat} {best next : Grid} : step m best next → 0 < m →
    LexLE best (m, m) → ∀ n, m ≤ n → NoWorse target next (m, n)

def referenceRow (target : Int) (m : Nat) (best next : Grid) : Prop :=
  RowRun target m m best next

def optimizedRow (target : Int) (m : Nat) (best next : Grid) : Prop :=
  next = fastRow target m best

theorem referenceRow_contract (target : Int) : RowContract target (referenceRow target) where
  terminates m positive best := row_exists target m positive m best
  improves run := run.improves
  origin run := run.origin
  minimizes run := run.minimizes

theorem optimizedRow_contract (target : Int) : RowContract target (optimizedRow target) where
  terminates m positive best := ⟨fastRow target m best, rfl⟩
  improves := by intro m best next same; rw [optimizedRow] at same; rw [same]; exact fastRow_improves _ _ _
  origin := by intro m best next same; rw [optimizedRow] at same; rw [same]; exact fastRow_origin _ _ _
  minimizes := by
    intro m best next same positive ordered n eligible
    rw [optimizedRow] at same
    rw [same]
    exact fastRow_minimizes target positive ordered n eligible

inductive RowsRun (target : Int) (row : Nat → Grid → Grid → Prop) : Nat → Grid → Grid → Prop
  | stop (m : Nat) (best : Grid) (finished : target + error target best < (count m m : Int)) :
      RowsRun target row m best best
  | step (m : Nat) (best next final : Grid)
      (active : (count m m : Int) ≤ target + error target best)
      (searched : row m best next) (tail : RowsRun target row (m + 1) next final) :
      RowsRun target row m best final

theorem RowsRun.improves {target : Int} {row : Nat → Grid → Grid → Prop}
    (contract : RowContract target row) {m : Nat} {best final : Grid}
    (run : RowsRun target row m best final) : NoWorse target final best := by
  induction run with
  | stop => exact noWorse_refl _ _
  | step m best next final active searched tail ih =>
      exact noWorse_trans ih (contract.improves searched)

theorem rows_exists (target : Int) (row : Nat → Grid → Grid → Prop)
    (contract : RowContract target row) (m : Nat) (positive : 0 < m) (best : Grid) :
    ∃ final, RowsRun target row m best final := by
  by_cases active : (count m m : Int) ≤ target + error target best
  · obtain ⟨next, searched⟩ := contract.terminates m positive best
    obtain ⟨final, tail⟩ := rows_exists target row contract (m + 1) (by omega) next
    exact ⟨final, .step m best next final active searched tail⟩
  · exact ⟨best, .stop m best (by omega)⟩
termination_by potential target m best
decreasing_by exact potential_decreases target positive active (noWorse_error (contract.improves searched))

theorem RowsRun.valid {target : Int} {row : Nat → Grid → Grid → Prop}
    (contract : RowContract target row) {m : Nat} {best final : Grid}
    (run : RowsRun target row m best final) : 0 < m → Valid best → Valid final := by
  induction run with
  | stop => exact fun _ valid => valid
  | step m best next final active searched tail ih =>
      intro positive initial
      apply ih (by omega)
      rcases contract.origin searched with same | ⟨first, second⟩
      · simpa only [same] using initial
      · unfold Valid
        omega

theorem RowsRun.minimizes {target : Int} {row : Nat → Grid → Grid → Prop}
    (contract : RowContract target row) {m : Nat} {best final : Grid}
    (run : RowsRun target row m best final) : 0 < m → LexLE best (m, m) →
      ∀ i n, m ≤ i → i ≤ n → NoWorse target final (i, n) := by
  induction run with
  | stop m best finished =>
      intro positive ordered i n later eligible
      exact Or.inl (stop_excludes_later target (error target best) finished later eligible)
  | step m best next final active searched tail ih =>
      intro positive ordered i n later eligible
      have nextOrdered : LexLE next (m + 1, m + 1) := by
        rcases contract.origin searched with same | ⟨first, second⟩
        · rw [same]
          unfold LexLE at *
          omega
        · unfold LexLE
          omega
      by_cases sameRow : i = m
      · subst i
        exact noWorse_trans (tail.improves contract) (contract.minimizes searched positive ordered n eligible)
      · exact ih (by omega) nextOrdered i n (by omega) eligible

def Optimal (target : Int) (grid : Grid) : Prop :=
  Valid grid ∧ ∀ other, Valid other → NoWorse target grid other

theorem optimal_unique {target : Int} {a b : Grid} (ha : Optimal target a) (hb : Optimal target b) :
    a = b := noWorse_antisymm (ha.2 b hb.1) (hb.2 a ha.1)

theorem RowsRun.optimal {target : Int} {row : Nat → Grid → Grid → Prop}
    (contract : RowContract target row) {final : Grid}
    (run : RowsRun target row 1 (1, 1) final) : Optimal target final := by
  refine ⟨run.valid contract (by omega) (by simp [Valid]), ?_⟩
  intro other valid
  exact run.minimizes contract (by omega) (lex_refl _) other.1 other.2 (by exact valid.1) valid.2

/-- Both complete searches terminate and return the same uniquely specified grid. -/
theorem searches_agree (target : Int) : ∃ final,
    RowsRun target (referenceRow target) 1 (1, 1) final ∧
    RowsRun target (optimizedRow target) 1 (1, 1) final ∧ Optimal target final := by
  obtain ⟨reference, hr⟩ := rows_exists target _ (referenceRow_contract target) 1 (by omega) (1, 1)
  obtain ⟨optimized, ho⟩ := rows_exists target _ (optimizedRow_contract target) 1 (by omega) (1, 1)
  have optimalR := hr.optimal (referenceRow_contract target)
  have optimalO := ho.optimal (optimizedRow_contract target)
  have same := optimal_unique optimalR optimalO
  exact ⟨reference, hr, same ▸ ho, optimalR⟩

end ProjectEulerSolutions.RectangleSearch
