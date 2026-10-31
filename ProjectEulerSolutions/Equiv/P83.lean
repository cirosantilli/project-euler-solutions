import ProjectEulerSolutions.Lib.DijkstraGridTraversal
namespace ProjectEulerSolutions.P83

theorem equiv (matrix : List (List Int))
    (rectangular : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value) :
    ProjectEulerStatements.P83.naive matrix = solve matrix := by
  by_cases rowsPositive : 0 < matrix.length
  · by_cases colsPositive : 0 < (matrix.headD []).length
    · obtain ⟨reference, referenceValue, referenceMinimum⟩ :=
        GridShortestPaths.naive_minimum matrix rectangular nonnegative rowsPositive colsPositive
      obtain ⟨optimized, optimizedValue, optimizedMinimum⟩ :=
        DijkstraGrid.solve_minimum matrix rectangular nonnegative rowsPositive colsPositive
      have lower := referenceMinimum optimized
      have upper := optimizedMinimum reference
      rw [referenceValue, optimizedValue] at lower upper
      exact le_antisymm lower upper
    · have emptyWidth : (matrix.headD []).length = 0 := by omega
      rw [GridRelaxationAgreement.naive_zero_width matrix emptyWidth]
      simp only [solve, dijkstraMinPathSum, emptyWidth, Nat.mul_zero, if_true]
  · have empty : matrix = [] := List.length_eq_zero_iff.mp (by omega)
    subst matrix
    rw [GridRelaxationAgreement.naive_empty]
    rfl
end ProjectEulerSolutions.P83
