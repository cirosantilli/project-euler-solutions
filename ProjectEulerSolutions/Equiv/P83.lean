import ProjectEulerSolutions.P83
namespace ProjectEulerSolutions.P83

theorem equiv (matrix : List (List Int))
    (_rectangular : ∀ row ∈ matrix, row.length = (matrix.headD []).length)
    (_nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value) : ProjectEulerStatements.P83.naive matrix = solve matrix := sorry
end ProjectEulerSolutions.P83
