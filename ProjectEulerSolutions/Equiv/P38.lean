import ProjectEulerSolutions.P38
namespace ProjectEulerSolutions.P38
open PandigitalConcat ProjectEulerStatements.P38

theorem findCount_correct (width seed count fuel : Nat) (positive : 0 < seed)
    (before : count ≤ stop width seed) (enough : stop width seed < count + fuel) :
    findCount width seed count fuel = stop width seed := by
  induction fuel generalizing count with
  | zero => omega
  | succ fuel ih =>
    rw [findCount]
    split
    · rename_i reached
      by_contra different
      have missing := before_stop width seed count positive (by omega)
      omega
    · rename_i missing
      have spec := stop_spec width seed positive
      have different : count ≠ stop width seed := by intro same; exact missing (same ▸ spec)
      apply ih (count+1) <;> omega

theorem seedScore_correct (width seed : Nat) (nonempty : 0 < width) (positive : 0 < seed) :
    seedScore width seed = PandigitalConcat.seedScore width seed := by
  unfold seedScore
  rw [findCount_correct width seed 0 (width+1) positive (Nat.zero_le _) (by
    have := (stop_bounds width seed nonempty positive).2
    omega)]
  rfl

/-- The reference enumerates all product counts; the solver stops at the first sufficient width. -/
theorem equiv (width : Nat) : reference width = solve width := by
  unfold solve
  split
  · rename_i outside
    rcases outside with small | large
    · exact reference_small width small
    · exact reference_large width large
  · rename_i inside
    have positive : 0 < width := by omega
    rw [reference_scan width positive]
    unfold scan
    have go (items : List Nat) (hp : ∀ n ∈ items, 0 < n) (initial : Nat) :
        items.foldl (fun best n => max best (PandigitalConcat.seedScore width n)) initial =
        items.foldl (fun best n => max best (seedScore width n)) initial := by
      induction items generalizing initial with
      | nil => rfl
      | cons head tail ih =>
        simp only [List.foldl_cons, seedScore_correct width head positive (hp head (by simp))]
        exact ih (fun n h => hp n (by simp [h])) _
    apply go
    intro n h
    have := List.mem_range'_1.mp h
    omega
end ProjectEulerSolutions.P38
