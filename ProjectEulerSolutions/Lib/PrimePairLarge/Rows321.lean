import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10529_small : List Nat := []
theorem row10529_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10529 row10529_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10529_block_005 : List Nat := []
theorem row10529_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10529 row10529_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10529_block_006 : List Nat := [10883, 11159]
def plan10529_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .second, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 23), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, .second, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 31), (.firstFactor 17), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 29), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10529_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10529 row10529_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10529_block_006
  · decide
  · rfl

def row10529_block_007 : List Nat := [12473, 12899, 13187, 13313, 13331]
def plan10529_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, .possible, .second, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .possible, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 29), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .possible, .second, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .first, .first, .possible, .first, .possible, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row10529_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10529 row10529_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10529_block_007
  · decide
  · rfl

def row10529_block_008 : List Nat := [14519, 14537, 14699, 15443]
def plan10529_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, .possible, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 37), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .second, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 29), (.firstFactor 13), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 13), (.secondFactor 7), .second, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10529_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10529 row10529_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10529_block_008
  · decide
  · rfl

def row10529_block_009 : List Nat := []
theorem row10529_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10529 row10529_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10529_block_010 : List Nat := []
theorem row10529_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10529 row10529_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10529_block_011 : List Nat := []
theorem row10529_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10529 row10529_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10529_block_012 : List Nat := []
theorem row10529_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10529 row10529_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10529_block_013 : List Nat := []
theorem row10529_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10529 row10529_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10529_large_left_left : List Nat := row10529_small ++ row10529_block_005
theorem row10529_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10529 row10529_large_left_left :=
  Budget.RowUpperBound.branch row10529_small_covers row10529_block_005_covers

def row10529_large_left_right_right : List Nat := row10529_block_007 ++ row10529_block_008
theorem row10529_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10529 row10529_large_left_right_right :=
  Budget.RowUpperBound.branch row10529_block_007_covers row10529_block_008_covers

def row10529_large_left_right : List Nat := row10529_block_006 ++ row10529_large_left_right_right
theorem row10529_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10529 row10529_large_left_right :=
  Budget.RowUpperBound.branch row10529_block_006_covers row10529_large_left_right_right_covers

def row10529_large_left : List Nat := row10529_large_left_left ++ row10529_large_left_right
theorem row10529_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10529 row10529_large_left :=
  Budget.RowUpperBound.branch row10529_large_left_left_covers row10529_large_left_right_covers

def row10529_large_right_left : List Nat := row10529_block_009 ++ row10529_block_010
theorem row10529_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10529 row10529_large_right_left :=
  Budget.RowUpperBound.branch row10529_block_009_covers row10529_block_010_covers

def row10529_large_right_right_right : List Nat := row10529_block_012 ++ row10529_block_013
theorem row10529_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10529 row10529_large_right_right_right :=
  Budget.RowUpperBound.branch row10529_block_012_covers row10529_block_013_covers

def row10529_large_right_right : List Nat := row10529_block_011 ++ row10529_large_right_right_right
theorem row10529_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10529 row10529_large_right_right :=
  Budget.RowUpperBound.branch row10529_block_011_covers row10529_large_right_right_right_covers

def row10529_large_right : List Nat := row10529_large_right_left ++ row10529_large_right_right
theorem row10529_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10529 row10529_large_right :=
  Budget.RowUpperBound.branch row10529_large_right_left_covers row10529_large_right_right_covers

def row10529_large : List Nat := row10529_large_left ++ row10529_large_right
theorem row10529_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10529 row10529_large :=
  Budget.RowUpperBound.branch row10529_large_left_covers row10529_large_right_covers

def row10529 : List Nat := row10529_large
theorem row10529_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10529 row10529 := row10529_large_covers

def row10531_small : List Nat := []
theorem row10531_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10531 row10531_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10531_block_005 : List Nat := []
theorem row10531_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10531 row10531_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10531_block_006 : List Nat := [11083, 11467, 12163, 12211]
def plan10531_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 37), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 11), .second, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 23), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .possible, .first, .first, (.firstFactor 37), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row10531_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10531 row10531_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10531_block_006
  · decide
  · rfl

def row10531_block_007 : List Nat := [12421, 13381, 14221, 14281]
def plan10531_block_007 : List Budget.Choice := [.possible, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 19), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .second, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 37), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, .second, (.secondFactor 17), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .possible, .first, .first, (.firstFactor 7), .possible, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10531_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10531 row10531_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10531_block_007
  · decide
  · rfl

def row10531_block_008 : List Nat := []
def plan10531_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .second, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), (.secondFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .first, .second, .first, .second, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10531_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10531 row10531_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10531_block_008
  · decide
  · rfl

def row10531_block_009 : List Nat := []
theorem row10531_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10531 row10531_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10531_block_010 : List Nat := []
theorem row10531_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10531 row10531_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10531_block_011 : List Nat := []
theorem row10531_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10531 row10531_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10531_block_012 : List Nat := []
theorem row10531_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10531 row10531_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10531_block_013 : List Nat := []
theorem row10531_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10531 row10531_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10531_large_left_left : List Nat := row10531_small ++ row10531_block_005
theorem row10531_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10531 row10531_large_left_left :=
  Budget.RowUpperBound.branch row10531_small_covers row10531_block_005_covers

def row10531_large_left_right_right : List Nat := row10531_block_007 ++ row10531_block_008
theorem row10531_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10531 row10531_large_left_right_right :=
  Budget.RowUpperBound.branch row10531_block_007_covers row10531_block_008_covers

def row10531_large_left_right : List Nat := row10531_block_006 ++ row10531_large_left_right_right
theorem row10531_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10531 row10531_large_left_right :=
  Budget.RowUpperBound.branch row10531_block_006_covers row10531_large_left_right_right_covers

def row10531_large_left : List Nat := row10531_large_left_left ++ row10531_large_left_right
theorem row10531_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10531 row10531_large_left :=
  Budget.RowUpperBound.branch row10531_large_left_left_covers row10531_large_left_right_covers

def row10531_large_right_left : List Nat := row10531_block_009 ++ row10531_block_010
theorem row10531_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10531 row10531_large_right_left :=
  Budget.RowUpperBound.branch row10531_block_009_covers row10531_block_010_covers

def row10531_large_right_right_right : List Nat := row10531_block_012 ++ row10531_block_013
theorem row10531_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10531 row10531_large_right_right_right :=
  Budget.RowUpperBound.branch row10531_block_012_covers row10531_block_013_covers

def row10531_large_right_right : List Nat := row10531_block_011 ++ row10531_large_right_right_right
theorem row10531_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10531 row10531_large_right_right :=
  Budget.RowUpperBound.branch row10531_block_011_covers row10531_large_right_right_right_covers

def row10531_large_right : List Nat := row10531_large_right_left ++ row10531_large_right_right
theorem row10531_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10531 row10531_large_right :=
  Budget.RowUpperBound.branch row10531_large_right_left_covers row10531_large_right_right_covers

def row10531_large : List Nat := row10531_large_left ++ row10531_large_right
theorem row10531_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10531 row10531_large :=
  Budget.RowUpperBound.branch row10531_large_left_covers row10531_large_right_covers

def row10531 : List Nat := row10531_large
theorem row10531_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10531 row10531 := row10531_large_covers

def row10559_small : List Nat := []
theorem row10559_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10559 row10559_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10559_block_005 : List Nat := []
theorem row10559_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10559 row10559_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10559_block_006 : List Nat := [10733, 11699, 11789, 11897, 12239]
def plan10559_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 29), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .possible, (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 37), .first, .first, (.secondFactor 31), (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .possible, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, .first]
set_option maxRecDepth 2048 in
theorem row10559_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10559 row10559_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10559_block_006
  · decide
  · rfl

def row10559_block_007 : List Nat := [12713, 13259, 13709, 14009]
def plan10559_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 19), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), (.secondFactor 17), (.firstFactor 11), (.firstFactor 13), .first, .first, .possible, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 31), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .second, .second, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 13), .second, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 29), .second, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .possible, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10559_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10559 row10559_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10559_block_007
  · decide
  · rfl

def row10559_block_008 : List Nat := [14879, 15173]
def plan10559_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 37), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, (.firstFactor 19), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .possible, .first, .first, .first, .first, (.secondFactor 19), .second, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 31), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10559_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10559 row10559_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10559_block_008
  · decide
  · rfl

def row10559_block_009 : List Nat := []
theorem row10559_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10559 row10559_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10559_block_010 : List Nat := []
theorem row10559_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10559 row10559_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10559_block_011 : List Nat := []
theorem row10559_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10559 row10559_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10559_block_012 : List Nat := []
theorem row10559_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10559 row10559_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10559_block_013 : List Nat := []
theorem row10559_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10559 row10559_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10559_large_left_left : List Nat := row10559_small ++ row10559_block_005
theorem row10559_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10559 row10559_large_left_left :=
  Budget.RowUpperBound.branch row10559_small_covers row10559_block_005_covers

def row10559_large_left_right_right : List Nat := row10559_block_007 ++ row10559_block_008
theorem row10559_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10559 row10559_large_left_right_right :=
  Budget.RowUpperBound.branch row10559_block_007_covers row10559_block_008_covers

def row10559_large_left_right : List Nat := row10559_block_006 ++ row10559_large_left_right_right
theorem row10559_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10559 row10559_large_left_right :=
  Budget.RowUpperBound.branch row10559_block_006_covers row10559_large_left_right_right_covers

def row10559_large_left : List Nat := row10559_large_left_left ++ row10559_large_left_right
theorem row10559_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10559 row10559_large_left :=
  Budget.RowUpperBound.branch row10559_large_left_left_covers row10559_large_left_right_covers

def row10559_large_right_left : List Nat := row10559_block_009 ++ row10559_block_010
theorem row10559_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10559 row10559_large_right_left :=
  Budget.RowUpperBound.branch row10559_block_009_covers row10559_block_010_covers

def row10559_large_right_right_right : List Nat := row10559_block_012 ++ row10559_block_013
theorem row10559_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10559 row10559_large_right_right_right :=
  Budget.RowUpperBound.branch row10559_block_012_covers row10559_block_013_covers

def row10559_large_right_right : List Nat := row10559_block_011 ++ row10559_large_right_right_right
theorem row10559_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10559 row10559_large_right_right :=
  Budget.RowUpperBound.branch row10559_block_011_covers row10559_large_right_right_right_covers

def row10559_large_right : List Nat := row10559_large_right_left ++ row10559_large_right_right
theorem row10559_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10559 row10559_large_right :=
  Budget.RowUpperBound.branch row10559_large_right_left_covers row10559_large_right_right_covers

def row10559_large : List Nat := row10559_large_left ++ row10559_large_right
theorem row10559_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10559 row10559_large :=
  Budget.RowUpperBound.branch row10559_large_left_covers row10559_large_right_covers

def row10559 : List Nat := row10559_large
theorem row10559_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10559 row10559 := row10559_large_covers

def row10567_small : List Nat := []
theorem row10567_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10567 row10567_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10567_block_005 : List Nat := []
theorem row10567_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10567 row10567_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10567_block_006 : List Nat := [12289]
def plan10567_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.secondFactor 17), .first, .first, .first, (.firstFactor 29), .second, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .second, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 37), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 29), .first, .first, (.firstFactor 23), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row10567_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10567 row10567_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10567_block_006
  · decide
  · rfl

def row10567_block_007 : List Nat := [12541, 13159, 13759, 14341]
def plan10567_block_007 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 23), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 19), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 37), (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 31), .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, .first, .second, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 31), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 13), .first, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row10567_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10567 row10567_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10567_block_007
  · decide
  · rfl

def row10567_block_008 : List Nat := [14479, 14851, 15091]
def plan10567_block_008 : List Budget.Choice := [.possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 31), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10567_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10567 row10567_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10567_block_008
  · decide
  · rfl

def row10567_block_009 : List Nat := []
theorem row10567_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10567 row10567_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10567_block_010 : List Nat := []
theorem row10567_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10567 row10567_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10567_block_011 : List Nat := []
theorem row10567_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10567 row10567_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10567_block_012 : List Nat := []
theorem row10567_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10567 row10567_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10567_block_013 : List Nat := []
theorem row10567_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10567 row10567_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10567_large_left_left : List Nat := row10567_small ++ row10567_block_005
theorem row10567_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10567 row10567_large_left_left :=
  Budget.RowUpperBound.branch row10567_small_covers row10567_block_005_covers

def row10567_large_left_right_right : List Nat := row10567_block_007 ++ row10567_block_008
theorem row10567_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10567 row10567_large_left_right_right :=
  Budget.RowUpperBound.branch row10567_block_007_covers row10567_block_008_covers

def row10567_large_left_right : List Nat := row10567_block_006 ++ row10567_large_left_right_right
theorem row10567_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10567 row10567_large_left_right :=
  Budget.RowUpperBound.branch row10567_block_006_covers row10567_large_left_right_right_covers

def row10567_large_left : List Nat := row10567_large_left_left ++ row10567_large_left_right
theorem row10567_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10567 row10567_large_left :=
  Budget.RowUpperBound.branch row10567_large_left_left_covers row10567_large_left_right_covers

def row10567_large_right_left : List Nat := row10567_block_009 ++ row10567_block_010
theorem row10567_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10567 row10567_large_right_left :=
  Budget.RowUpperBound.branch row10567_block_009_covers row10567_block_010_covers

def row10567_large_right_right_right : List Nat := row10567_block_012 ++ row10567_block_013
theorem row10567_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10567 row10567_large_right_right_right :=
  Budget.RowUpperBound.branch row10567_block_012_covers row10567_block_013_covers

def row10567_large_right_right : List Nat := row10567_block_011 ++ row10567_large_right_right_right
theorem row10567_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10567 row10567_large_right_right :=
  Budget.RowUpperBound.branch row10567_block_011_covers row10567_large_right_right_right_covers

def row10567_large_right : List Nat := row10567_large_right_left ++ row10567_large_right_right
theorem row10567_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10567 row10567_large_right :=
  Budget.RowUpperBound.branch row10567_large_right_left_covers row10567_large_right_right_covers

def row10567_large : List Nat := row10567_large_left ++ row10567_large_right
theorem row10567_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10567 row10567_large :=
  Budget.RowUpperBound.branch row10567_large_left_covers row10567_large_right_covers

def row10567 : List Nat := row10567_large
theorem row10567_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10567 row10567 := row10567_large_covers

end ProjectEulerSolutions.PrimePairLarge
