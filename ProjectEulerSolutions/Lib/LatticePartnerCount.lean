import ProjectEulerSolutions.Lib.LatticeStepBounds

/-! Counts of nondegenerate right triangles with a fixed right-angle vertex. -/
namespace ProjectEulerSolutions.LatticeRightTriangles
set_option Elab.async false

def partners (n : Nat) (p : Int × Int) : Finset (Int × Int) :=
  (grid n).filter fun q => p ≠ (0, 0) ∧ q ≠ (0, 0) ∧ q ≠ p ∧ atFirst p.1 p.2 q.1 q.2

theorem mem_partners (n : Nat) (p q : Int × Int) :
    q ∈ partners n p ↔ q ∈ grid n ∧ p ≠ (0, 0) ∧ q ≠ (0, 0) ∧ q ≠ p ∧ atFirst p.1 p.2 q.1 q.2 := by
  simp [partners]

theorem partner_nonzero (p q : Int × Int) (nonzero : p ≠ (0, 0))
    (angle : atFirst p.1 p.2 q.1 q.2) : q ≠ (0, 0) := by
  rintro rfl
  have zero : norm p.1 p.2 = 0 := by
    dsimp [atFirst] at angle
    dsimp [norm]
    nlinarith
  obtain ⟨hx, hy⟩ := (norm_zero _ _).mp zero
  exact nonzero (Prod.ext hx hy)

/-- The signed primitive steps are a bijection with all partners of an interior point. -/
theorem interior_partners_card (n x y : Nat) (hx : 0 < x) (hy : 0 < y)
    (xBound : x ≤ n) (yBound : y ≤ n) :
    (partners n ((x : Int), (y : Int))).card = upperSteps n x y + lowerSteps n x y := by
  have nonzero : ((x : Int), (y : Int)) ≠ (0, 0) := by
    intro same
    have := congrArg Prod.fst same
    dsimp at this
    omega
  have counted : (admissibleSteps (lowerSteps n x y) (upperSteps n x y)).card =
      (partners n ((x : Int), (y : Int))).card := by
    apply Finset.card_bij (fun k _ => stepPoint x y k)
    · intro k member
      rcases (mem_admissibleSteps _ _ _).mp member with ⟨notZero, low, high⟩
      have angle : atFirst (x : Int) (y : Int) (stepPoint x y k).1 (stepPoint x y k).2 :=
        (perpendicular_points x y hx _ _).mpr ⟨k, rfl, rfl⟩
      exact (mem_partners _ _ _).mpr
        ⟨(stepPoint_in_grid n x y hx hy xBound yBound k).mpr ⟨low, high⟩,
          nonzero, partner_nonzero _ _ nonzero angle,
          fun same => notZero ((stepPoint_eq_self x y hx k).mp same), angle⟩
    · intro k _ l _ same
      exact stepPoint_injective x y hx same
    · intro q member
      rcases (mem_partners _ _ _).mp member with ⟨inside, _, _, distinct, angle⟩
      obtain ⟨k, first, second⟩ := (perpendicular_points x y hx q.1 q.2).mp angle
      have same : stepPoint x y k = q := Prod.ext first.symm second.symm
      have interval := (stepPoint_in_grid n x y hx hy xBound yBound k).mp (same.symm ▸ inside)
      refine ⟨k, (mem_admissibleSteps _ _ _).mpr ⟨?_, interval⟩, same⟩
      intro zero
      apply distinct
      rw [← same, (stepPoint_eq_self x y hx k).mpr zero]
  rw [admissibleSteps_card] at counted
  omega

theorem partners_origin (n : Nat) : partners n (0, 0) = ∅ := by
  ext q
  simp [partners]

theorem partners_swap_card (n : Nat) (p : Int × Int) :
    (partners n p.swap).card = (partners n p).card := by
  apply Finset.card_bij (fun q _ => q.swap)
  · intro q member
    rcases (mem_partners _ _ _).mp member with ⟨inside, nonzero, qNonzero, distinct, angle⟩
    refine (mem_partners _ _ _).mpr ⟨?_, ?_, ?_, ?_, ?_⟩
    · simpa only [mem_grid, Prod.fst_swap, Prod.snd_swap, and_assoc, and_left_comm, and_comm] using inside
    · intro same
      apply nonzero
      simpa using congrArg Prod.swap same
    · intro same
      apply qNonzero
      simpa using congrArg Prod.swap same
    · intro same
      apply distinct
      simpa using congrArg Prod.swap same
    · simpa only [atFirst, Prod.fst_swap, Prod.snd_swap, add_comm] using angle
  · intro q _ r _ same
    exact Prod.swap_injective same
  · intro q member
    refine ⟨q.swap, ?_, Prod.swap_swap q⟩
    rcases (mem_partners _ _ _).mp member with ⟨inside, nonzero, qNonzero, distinct, angle⟩
    refine (mem_partners _ _ _).mpr ⟨?_, ?_, ?_, ?_, ?_⟩
    · simpa only [mem_grid, Prod.fst_swap, Prod.snd_swap, and_assoc, and_left_comm, and_comm] using inside
    · intro same
      apply nonzero
      simpa using congrArg Prod.swap same
    · intro same
      apply qNonzero
      simpa using congrArg Prod.swap same
    · intro same
      apply distinct
      exact Prod.swap_injective same
    · simpa only [atFirst, Prod.fst_swap, Prod.snd_swap, add_comm] using angle

/-- A nonzero point on the horizontal axis has exactly one partner at each positive height. -/
theorem horizontal_partners_card (n x : Nat) (positive : 0 < x) (bounded : x ≤ n) :
    (partners n ((x : Int), 0)).card = n := by
  have counted : (Finset.Icc (1 : Int) (n : Int)).card = (partners n ((x : Int), 0)).card := by
    apply Finset.card_bij (fun y _ => ((x : Int), y))
    · intro y member
      have bounds := Finset.mem_Icc.mp member
      refine (mem_partners _ _ _).mpr ⟨?_, ?_, ?_, ?_, ?_⟩
      · rw [mem_grid]
        dsimp
        exact ⟨by omega, by omega, by omega, bounds.2⟩
      · intro same
        have := congrArg Prod.fst same
        dsimp at this
        omega
      · intro same
        have := congrArg Prod.snd same
        dsimp at this
        omega
      · intro same
        have := congrArg Prod.snd same
        dsimp at this
        omega
      · simp [atFirst]
    · intro y _ z _ same
      exact congrArg Prod.snd same
    · intro q member
      rcases (mem_partners _ _ _).mp member with ⟨inside, _, _, distinct, angle⟩
      have bounds := (mem_grid n q).mp inside
      have first : q.1 = (x : Int) := by
        dsimp [atFirst] at angle
        have xpos : (0 : Int) < x := by exact_mod_cast positive
        nlinarith
      have second : 1 ≤ q.2 := by
        by_contra no
        have zero : q.2 = 0 := by omega
        exact distinct (Prod.ext first zero)
      exact ⟨q.2, Finset.mem_Icc.mpr ⟨second, bounds.2.2.2⟩, Prod.ext first.symm rfl⟩
  simpa only [Int.card_Icc, add_sub_cancel_right, Int.toNat_natCast] using counted.symm

theorem vertical_partners_card (n y : Nat) (positive : 0 < y) (bounded : y ≤ n) :
    (partners n (0, (y : Int))).card = n := by
  simpa using (partners_swap_card n ((y : Int), 0)).trans (horizontal_partners_card n y positive bounded)

def pointContribution (n x y : Nat) : Nat :=
  if x = 0 ∧ y = 0 then 0 else if x = 0 ∨ y = 0 then n else upperSteps n x y + lowerSteps n x y

/-- The source's origin, axis and interior branches count exactly the eligible partners. -/
theorem partners_card (n x y : Nat) (xBound : x ≤ n) (yBound : y ≤ n) :
    (partners n ((x : Int), (y : Int))).card = pointContribution n x y := by
  by_cases hx : x = 0
  · subst x
    by_cases hy : y = 0
    · subst y
      simp [pointContribution, partners_origin]
    · simpa [pointContribution, hy] using vertical_partners_card n y (by omega) yBound
  · by_cases hy : y = 0
    · subst y
      simpa [pointContribution, hx] using horizontal_partners_card n x (by omega) xBound
    · simpa [pointContribution, hx, hy] using interior_partners_card n x y (by omega) (by omega) xBound yBound

end ProjectEulerSolutions.LatticeRightTriangles
