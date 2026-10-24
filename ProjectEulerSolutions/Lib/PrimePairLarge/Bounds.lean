import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairBudgetSearch
namespace ProjectEulerSolutions.PrimePairLarge.Bounds
set_option maxRecDepth 4096 in
theorem block_000 : ∀ q ∈ PrimePairVertices.tree_block_000.values, 2 ≤ q ∧ q < 2050 := by decide
set_option maxRecDepth 4096 in
theorem block_001 : ∀ q ∈ PrimePairVertices.tree_block_001.values, 2050 ≤ q ∧ q < 4098 := by decide
set_option maxRecDepth 4096 in
theorem block_002 : ∀ q ∈ PrimePairVertices.tree_block_002.values, 4098 ≤ q ∧ q < 6146 := by decide
set_option maxRecDepth 4096 in
theorem block_003 : ∀ q ∈ PrimePairVertices.tree_block_003.values, 6146 ≤ q ∧ q < 8194 := by decide
set_option maxRecDepth 4096 in
theorem block_004 : ∀ q ∈ PrimePairVertices.tree_block_004.values, 8194 ≤ q ∧ q < 8389 := by decide
set_option maxRecDepth 4096 in
theorem block_005 : ∀ q ∈ PrimePairVertices.tree_block_005.values, 8389 ≤ q ∧ q < 10370 := by decide
set_option maxRecDepth 4096 in
theorem block_006 : ∀ q ∈ PrimePairVertices.tree_block_006.values, 10370 ≤ q ∧ q < 12418 := by decide
set_option maxRecDepth 4096 in
theorem block_007 : ∀ q ∈ PrimePairVertices.tree_block_007.values, 12418 ≤ q ∧ q < 14466 := by decide
set_option maxRecDepth 4096 in
theorem block_008 : ∀ q ∈ PrimePairVertices.tree_block_008.values, 14466 ≤ q ∧ q < 16514 := by decide
set_option maxRecDepth 4096 in
theorem block_009 : ∀ q ∈ PrimePairVertices.tree_block_009.values, 16514 ≤ q ∧ q < 18562 := by decide
set_option maxRecDepth 4096 in
theorem block_010 : ∀ q ∈ PrimePairVertices.tree_block_010.values, 18562 ≤ q ∧ q < 20610 := by decide
set_option maxRecDepth 4096 in
theorem block_011 : ∀ q ∈ PrimePairVertices.tree_block_011.values, 20610 ≤ q ∧ q < 22658 := by decide
set_option maxRecDepth 4096 in
theorem block_012 : ∀ q ∈ PrimePairVertices.tree_block_012.values, 22658 ≤ q ∧ q < 24706 := by decide
set_option maxRecDepth 4096 in
theorem block_013 : ∀ q ∈ PrimePairVertices.tree_block_013.values, 24706 ≤ q ∧ q < 26034 := by decide
theorem small : ∀ q ∈ PrimePairVertices.tree_small.values, 2 ≤ q ∧ q < 8389 := by
  intro q member
  have shape : PrimePairVertices.tree_small.values =
      (PrimePairVertices.tree_block_000.values ++ PrimePairVertices.tree_block_001.values) ++
        (PrimePairVertices.tree_block_002.values ++ (PrimePairVertices.tree_block_003.values ++ PrimePairVertices.tree_block_004.values)) := rfl
  rw [shape] at member
  simp only [List.mem_append] at member
  rcases member with (first | second) | third | fourth | fifth
  · have := block_000 q first; omega
  · have := block_001 q second; omega
  · have := block_002 q third; omega
  · have := block_003 q fourth; omega
  · have := block_004 q fifth; omega
end ProjectEulerSolutions.PrimePairLarge.Bounds
