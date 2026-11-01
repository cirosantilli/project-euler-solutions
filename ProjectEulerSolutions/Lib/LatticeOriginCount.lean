import ProjectEulerSolutions.Lib.LatticePartnerCount

/-! Opposite positive axes count the right angles at the distinguished origin. -/
namespace ProjectEulerSolutions.LatticeRightTriangles
set_option Elab.async false

def nondegenerate (p q : Int × Int) : Prop := p ≠ (0, 0) ∧ q ≠ (0, 0) ∧ q ≠ p
instance (p q : Int × Int) : Decidable (nondegenerate p q) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def xAxis (n : Nat) : Finset (Int × Int) := (Finset.Icc (1 : Int) (n : Int)).image fun x => (x, 0)
def yAxis (n : Nat) : Finset (Int × Int) := (Finset.Icc (1 : Int) (n : Int)).image fun y => (0, y)

theorem mem_xAxis (n : Nat) (p : Int × Int) : p ∈ xAxis n ↔ 1 ≤ p.1 ∧ p.1 ≤ n ∧ p.2 = 0 := by
  simp only [xAxis, Finset.mem_image, Finset.mem_Icc, Prod.ext_iff]
  constructor
  · rintro ⟨x, hx, first, second⟩
    exact ⟨first ▸ hx.1, first ▸ hx.2, second.symm⟩
  · rintro ⟨low, high, zero⟩
    exact ⟨p.1, ⟨low, high⟩, rfl, zero.symm⟩

theorem mem_yAxis (n : Nat) (p : Int × Int) : p ∈ yAxis n ↔ p.1 = 0 ∧ 1 ≤ p.2 ∧ p.2 ≤ n := by
  simp only [yAxis, Finset.mem_image, Finset.mem_Icc, Prod.ext_iff]
  constructor
  · rintro ⟨y, hy, first, second⟩
    exact ⟨first.symm, second ▸ hy.1, second ▸ hy.2⟩
  · rintro ⟨zero, low, high⟩
    exact ⟨p.2, ⟨low, high⟩, zero.symm, rfl⟩

theorem xAxis_card (n : Nat) : (xAxis n).card = n := by
  rw [xAxis, Finset.card_image_of_injective _ (fun a b same => congrArg Prod.fst same), Int.card_Icc]
  simp

theorem yAxis_card (n : Nat) : (yAxis n).card = n := by
  rw [yAxis, Finset.card_image_of_injective _ (fun a b same => congrArg Prod.snd same), Int.card_Icc]
  simp

def originPairs (n : Nat) : Finset ((Int × Int) × (Int × Int)) :=
  ((grid n) ×ˢ (grid n)).filter fun pair =>
    nondegenerate pair.1 pair.2 ∧ atOrigin pair.1.1 pair.1.2 pair.2.1 pair.2.2

theorem origin_axes (n : Nat) (p q : Int × Int) :
    (p, q) ∈ originPairs n ↔ p ∈ xAxis n ∧ q ∈ yAxis n ∨ p ∈ yAxis n ∧ q ∈ xAxis n := by
  simp only [originPairs, Finset.mem_filter, Finset.mem_product, mem_grid, mem_xAxis, mem_yAxis]
  constructor
  · rintro ⟨⟨hp, hq⟩, ⟨pn, qn, _⟩, angle⟩
    have zeroX : p.1 * q.1 = 0 := by
      dsimp [atOrigin] at angle
      nlinarith [mul_nonneg hp.1 hq.1, mul_nonneg hp.2.2.1 hq.2.2.1]
    have zeroY : p.2 * q.2 = 0 := by
      dsimp [atOrigin] at angle
      nlinarith [mul_nonneg hp.1 hq.1, mul_nonneg hp.2.2.1 hq.2.2.1]
    by_cases px : p.1 = 0
    · have py : p.2 ≠ 0 := fun hy => pn (Prod.ext px hy)
      have qy : q.2 = 0 := (mul_eq_zero.mp zeroY).resolve_left py
      have qx : q.1 ≠ 0 := fun hx => qn (Prod.ext hx qy)
      right
      exact ⟨⟨px, by omega, hp.2.2.2⟩, ⟨by omega, hq.2.1, qy⟩⟩
    · have qx : q.1 = 0 := (mul_eq_zero.mp zeroX).resolve_left px
      have qy : q.2 ≠ 0 := fun hy => qn (Prod.ext qx hy)
      have py : p.2 = 0 := (mul_eq_zero.mp zeroY).resolve_right qy
      left
      exact ⟨⟨by omega, hp.2.1, py⟩, ⟨qx, by omega, hq.2.2.2⟩⟩
  · rintro (⟨⟨pl, ph, py⟩, ⟨qx, ql, qh⟩⟩ | ⟨⟨px, pl, ph⟩, ⟨ql, qh, qy⟩⟩)
    · refine ⟨⟨⟨by omega, ph, by omega, by omega⟩, ⟨by omega, by omega, by omega, qh⟩⟩, ?_, ?_⟩
      · refine ⟨?_, ?_, ?_⟩
        · intro same
          have := congrArg Prod.fst same
          have := congrArg Prod.snd same
          dsimp at *
          omega
        · intro same
          have := congrArg Prod.fst same
          have := congrArg Prod.snd same
          dsimp at *
          omega
        · intro same
          have := congrArg Prod.fst same
          have := congrArg Prod.snd same
          omega
      · simp [atOrigin, py, qx]
    · refine ⟨⟨⟨by omega, by omega, by omega, ph⟩, ⟨by omega, qh, by omega, by omega⟩⟩, ?_, ?_⟩
      · refine ⟨?_, ?_, ?_⟩
        · intro same
          have := congrArg Prod.fst same
          have := congrArg Prod.snd same
          dsimp at *
          omega
        · intro same
          have := congrArg Prod.fst same
          have := congrArg Prod.snd same
          dsimp at *
          omega
        · intro same
          have := congrArg Prod.fst same
          have := congrArg Prod.snd same
          omega
      · simp [atOrigin, px, qy]

/-- There are n squared triangles at the origin, and two orders of their other vertices. -/
theorem originPairs_card (n : Nat) : (originPairs n).card = 2 * (n * n) := by
  have sets : originPairs n = (xAxis n ×ˢ yAxis n) ∪ (yAxis n ×ˢ xAxis n) := by
    ext pair
    simpa only [Finset.mem_union, Finset.mem_product] using origin_axes n pair.1 pair.2
  have separate : Disjoint (xAxis n ×ˢ yAxis n) (yAxis n ×ˢ xAxis n) := by
    rw [Finset.disjoint_left]
    intro pair left right
    have px := (mem_xAxis n pair.1).mp (Finset.mem_product.mp left).1
    have py := (mem_yAxis n pair.1).mp (Finset.mem_product.mp right).1
    omega
  rw [sets, Finset.card_union_of_disjoint separate, Finset.card_product, Finset.card_product,
    xAxis_card, yAxis_card]
  omega

end ProjectEulerSolutions.LatticeRightTriangles
