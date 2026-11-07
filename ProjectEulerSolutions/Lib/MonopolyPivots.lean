import ProjectEulerSolutions.Lib.MonopolySystem
import ProjectEulerSolutions.Lib.GaussianPivot

namespace ProjectEulerSolutions.Monopoly
set_option Elab.async false

/-- The first source pivot search has a nonzero candidate for every positive die size. -/
theorem initial_pivot (sides : Nat) (positive : 0 < sides) :
    ∃ row : Fin 120, coefficients sides row 0 ≠ 0 := by
  obtain ⟨row, _, nonzero⟩ := GaussianPivot.exists_pivot_of_injective
    (fun row column => (coefficients sides row column : ℚ)) (coefficients_bijective sides positive).injective 0
    (by intro row previous earlier _; simp at earlier)
  exact ⟨row, by exact_mod_cast nonzero⟩

end ProjectEulerSolutions.Monopoly
