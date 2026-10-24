import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12613_small : List Nat := []
theorem row12613_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12613 row12613_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12613_block_005 : List Nat := []
theorem row12613_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12613 row12613_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12613_block_006 : List Nat := []
theorem row12613_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12613 row12613_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12613_block_007 : List Nat := [13267]
def plan12613_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 17), .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 7), (.secondFactor 17), (.firstFactor 19), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 23), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12613_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12613 row12613_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12613_block_007
  · decide
  · rfl

def row12613_block_008 : List Nat := []
theorem row12613_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12613 row12613_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12613_block_009 : List Nat := []
theorem row12613_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12613 row12613_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12613_block_010 : List Nat := []
theorem row12613_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12613 row12613_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12613_block_011 : List Nat := []
theorem row12613_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12613 row12613_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12613_block_012 : List Nat := []
theorem row12613_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12613 row12613_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12613_block_013 : List Nat := []
theorem row12613_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12613 row12613_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12613_large_left_left : List Nat := row12613_small ++ row12613_block_005
theorem row12613_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12613 row12613_large_left_left :=
  Budget.RowUpperBound.branch row12613_small_covers row12613_block_005_covers

def row12613_large_left_right_right : List Nat := row12613_block_007 ++ row12613_block_008
theorem row12613_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12613 row12613_large_left_right_right :=
  Budget.RowUpperBound.branch row12613_block_007_covers row12613_block_008_covers

def row12613_large_left_right : List Nat := row12613_block_006 ++ row12613_large_left_right_right
theorem row12613_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12613 row12613_large_left_right :=
  Budget.RowUpperBound.branch row12613_block_006_covers row12613_large_left_right_right_covers

def row12613_large_left : List Nat := row12613_large_left_left ++ row12613_large_left_right
theorem row12613_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12613 row12613_large_left :=
  Budget.RowUpperBound.branch row12613_large_left_left_covers row12613_large_left_right_covers

def row12613_large_right_left : List Nat := row12613_block_009 ++ row12613_block_010
theorem row12613_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12613 row12613_large_right_left :=
  Budget.RowUpperBound.branch row12613_block_009_covers row12613_block_010_covers

def row12613_large_right_right_right : List Nat := row12613_block_012 ++ row12613_block_013
theorem row12613_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12613 row12613_large_right_right_right :=
  Budget.RowUpperBound.branch row12613_block_012_covers row12613_block_013_covers

def row12613_large_right_right : List Nat := row12613_block_011 ++ row12613_large_right_right_right
theorem row12613_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12613 row12613_large_right_right :=
  Budget.RowUpperBound.branch row12613_block_011_covers row12613_large_right_right_right_covers

def row12613_large_right : List Nat := row12613_large_right_left ++ row12613_large_right_right
theorem row12613_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12613 row12613_large_right :=
  Budget.RowUpperBound.branch row12613_large_right_left_covers row12613_large_right_right_covers

def row12613_large : List Nat := row12613_large_left ++ row12613_large_right
theorem row12613_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12613 row12613_large :=
  Budget.RowUpperBound.branch row12613_large_left_covers row12613_large_right_covers

def row12613 : List Nat := row12613_large
theorem row12613_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12613 row12613 := row12613_large_covers

def row12619_small : List Nat := []
theorem row12619_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12619 row12619_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12619_block_005 : List Nat := []
theorem row12619_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12619 row12619_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12619_block_006 : List Nat := []
theorem row12619_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12619 row12619_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12619_block_007 : List Nat := [12829]
def plan12619_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 31), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .second, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 29), (.firstFactor 19), .second, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12619_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12619 row12619_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12619_block_007
  · decide
  · rfl

def row12619_block_008 : List Nat := []
theorem row12619_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12619 row12619_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12619_block_009 : List Nat := []
theorem row12619_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12619 row12619_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12619_block_010 : List Nat := []
theorem row12619_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12619 row12619_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12619_block_011 : List Nat := []
theorem row12619_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12619 row12619_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12619_block_012 : List Nat := []
theorem row12619_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12619 row12619_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12619_block_013 : List Nat := []
theorem row12619_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12619 row12619_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12619_large_left_left : List Nat := row12619_small ++ row12619_block_005
theorem row12619_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12619 row12619_large_left_left :=
  Budget.RowUpperBound.branch row12619_small_covers row12619_block_005_covers

def row12619_large_left_right_right : List Nat := row12619_block_007 ++ row12619_block_008
theorem row12619_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12619 row12619_large_left_right_right :=
  Budget.RowUpperBound.branch row12619_block_007_covers row12619_block_008_covers

def row12619_large_left_right : List Nat := row12619_block_006 ++ row12619_large_left_right_right
theorem row12619_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12619 row12619_large_left_right :=
  Budget.RowUpperBound.branch row12619_block_006_covers row12619_large_left_right_right_covers

def row12619_large_left : List Nat := row12619_large_left_left ++ row12619_large_left_right
theorem row12619_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12619 row12619_large_left :=
  Budget.RowUpperBound.branch row12619_large_left_left_covers row12619_large_left_right_covers

def row12619_large_right_left : List Nat := row12619_block_009 ++ row12619_block_010
theorem row12619_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12619 row12619_large_right_left :=
  Budget.RowUpperBound.branch row12619_block_009_covers row12619_block_010_covers

def row12619_large_right_right_right : List Nat := row12619_block_012 ++ row12619_block_013
theorem row12619_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12619 row12619_large_right_right_right :=
  Budget.RowUpperBound.branch row12619_block_012_covers row12619_block_013_covers

def row12619_large_right_right : List Nat := row12619_block_011 ++ row12619_large_right_right_right
theorem row12619_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12619 row12619_large_right_right :=
  Budget.RowUpperBound.branch row12619_block_011_covers row12619_large_right_right_right_covers

def row12619_large_right : List Nat := row12619_large_right_left ++ row12619_large_right_right
theorem row12619_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12619 row12619_large_right :=
  Budget.RowUpperBound.branch row12619_large_right_left_covers row12619_large_right_right_covers

def row12619_large : List Nat := row12619_large_left ++ row12619_large_right
theorem row12619_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12619 row12619_large :=
  Budget.RowUpperBound.branch row12619_large_left_covers row12619_large_right_covers

def row12619 : List Nat := row12619_large
theorem row12619_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12619 row12619 := row12619_large_covers

def row12637_small : List Nat := []
theorem row12637_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12637 row12637_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12637_block_005 : List Nat := []
theorem row12637_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12637 row12637_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12637_block_006 : List Nat := []
theorem row12637_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12637 row12637_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12637_block_007 : List Nat := [12841]
def plan12637_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 37), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, .possible, (.firstFactor 13), (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12637_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12637 row12637_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12637_block_007
  · decide
  · rfl

def row12637_block_008 : List Nat := []
theorem row12637_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12637 row12637_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12637_block_009 : List Nat := []
theorem row12637_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12637 row12637_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12637_block_010 : List Nat := []
theorem row12637_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12637 row12637_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12637_block_011 : List Nat := []
theorem row12637_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12637 row12637_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12637_block_012 : List Nat := []
theorem row12637_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12637 row12637_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12637_block_013 : List Nat := []
theorem row12637_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12637 row12637_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12637_large_left_left : List Nat := row12637_small ++ row12637_block_005
theorem row12637_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12637 row12637_large_left_left :=
  Budget.RowUpperBound.branch row12637_small_covers row12637_block_005_covers

def row12637_large_left_right_right : List Nat := row12637_block_007 ++ row12637_block_008
theorem row12637_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12637 row12637_large_left_right_right :=
  Budget.RowUpperBound.branch row12637_block_007_covers row12637_block_008_covers

def row12637_large_left_right : List Nat := row12637_block_006 ++ row12637_large_left_right_right
theorem row12637_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12637 row12637_large_left_right :=
  Budget.RowUpperBound.branch row12637_block_006_covers row12637_large_left_right_right_covers

def row12637_large_left : List Nat := row12637_large_left_left ++ row12637_large_left_right
theorem row12637_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12637 row12637_large_left :=
  Budget.RowUpperBound.branch row12637_large_left_left_covers row12637_large_left_right_covers

def row12637_large_right_left : List Nat := row12637_block_009 ++ row12637_block_010
theorem row12637_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12637 row12637_large_right_left :=
  Budget.RowUpperBound.branch row12637_block_009_covers row12637_block_010_covers

def row12637_large_right_right_right : List Nat := row12637_block_012 ++ row12637_block_013
theorem row12637_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12637 row12637_large_right_right_right :=
  Budget.RowUpperBound.branch row12637_block_012_covers row12637_block_013_covers

def row12637_large_right_right : List Nat := row12637_block_011 ++ row12637_large_right_right_right
theorem row12637_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12637 row12637_large_right_right :=
  Budget.RowUpperBound.branch row12637_block_011_covers row12637_large_right_right_right_covers

def row12637_large_right : List Nat := row12637_large_right_left ++ row12637_large_right_right
theorem row12637_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12637 row12637_large_right :=
  Budget.RowUpperBound.branch row12637_large_right_left_covers row12637_large_right_right_covers

def row12637_large : List Nat := row12637_large_left ++ row12637_large_right
theorem row12637_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12637 row12637_large :=
  Budget.RowUpperBound.branch row12637_large_left_covers row12637_large_right_covers

def row12637 : List Nat := row12637_large
theorem row12637_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12637 row12637 := row12637_large_covers

def row12641_small : List Nat := []
theorem row12641_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12641 row12641_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12641_block_005 : List Nat := []
theorem row12641_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12641 row12641_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12641_block_006 : List Nat := []
theorem row12641_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12641 row12641_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12641_block_007 : List Nat := []
def plan12641_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .second, (.firstFactor 37), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12641_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12641 row12641_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12641_block_007
  · decide
  · rfl

def row12641_block_008 : List Nat := []
theorem row12641_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12641 row12641_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12641_block_009 : List Nat := []
theorem row12641_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12641 row12641_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12641_block_010 : List Nat := []
theorem row12641_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12641 row12641_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12641_block_011 : List Nat := []
theorem row12641_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12641 row12641_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12641_block_012 : List Nat := []
theorem row12641_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12641 row12641_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12641_block_013 : List Nat := []
theorem row12641_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12641 row12641_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12641_large_left_left : List Nat := row12641_small ++ row12641_block_005
theorem row12641_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12641 row12641_large_left_left :=
  Budget.RowUpperBound.branch row12641_small_covers row12641_block_005_covers

def row12641_large_left_right_right : List Nat := row12641_block_007 ++ row12641_block_008
theorem row12641_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12641 row12641_large_left_right_right :=
  Budget.RowUpperBound.branch row12641_block_007_covers row12641_block_008_covers

def row12641_large_left_right : List Nat := row12641_block_006 ++ row12641_large_left_right_right
theorem row12641_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12641 row12641_large_left_right :=
  Budget.RowUpperBound.branch row12641_block_006_covers row12641_large_left_right_right_covers

def row12641_large_left : List Nat := row12641_large_left_left ++ row12641_large_left_right
theorem row12641_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12641 row12641_large_left :=
  Budget.RowUpperBound.branch row12641_large_left_left_covers row12641_large_left_right_covers

def row12641_large_right_left : List Nat := row12641_block_009 ++ row12641_block_010
theorem row12641_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12641 row12641_large_right_left :=
  Budget.RowUpperBound.branch row12641_block_009_covers row12641_block_010_covers

def row12641_large_right_right_right : List Nat := row12641_block_012 ++ row12641_block_013
theorem row12641_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12641 row12641_large_right_right_right :=
  Budget.RowUpperBound.branch row12641_block_012_covers row12641_block_013_covers

def row12641_large_right_right : List Nat := row12641_block_011 ++ row12641_large_right_right_right
theorem row12641_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12641 row12641_large_right_right :=
  Budget.RowUpperBound.branch row12641_block_011_covers row12641_large_right_right_right_covers

def row12641_large_right : List Nat := row12641_large_right_left ++ row12641_large_right_right
theorem row12641_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12641 row12641_large_right :=
  Budget.RowUpperBound.branch row12641_large_right_left_covers row12641_large_right_right_covers

def row12641_large : List Nat := row12641_large_left ++ row12641_large_right
theorem row12641_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12641 row12641_large :=
  Budget.RowUpperBound.branch row12641_large_left_covers row12641_large_right_covers

def row12641 : List Nat := row12641_large
theorem row12641_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12641 row12641 := row12641_large_covers

end ProjectEulerSolutions.PrimePairLarge
