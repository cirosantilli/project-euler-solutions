import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12809_small : List Nat := []
theorem row12809_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12809 row12809_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12809_block_005 : List Nat := []
theorem row12809_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12809 row12809_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12809_block_006 : List Nat := []
theorem row12809_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12809 row12809_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12809_block_007 : List Nat := []
def plan12809_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 29), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 37), (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12809_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12809 row12809_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12809_block_007
  · decide
  · rfl

def row12809_block_008 : List Nat := []
theorem row12809_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12809 row12809_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12809_block_009 : List Nat := []
theorem row12809_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12809 row12809_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12809_block_010 : List Nat := []
theorem row12809_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12809 row12809_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12809_block_011 : List Nat := []
theorem row12809_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12809 row12809_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12809_block_012 : List Nat := []
theorem row12809_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12809 row12809_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12809_block_013 : List Nat := []
theorem row12809_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12809 row12809_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12809_large_left_left : List Nat := row12809_small ++ row12809_block_005
theorem row12809_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12809 row12809_large_left_left :=
  Budget.RowUpperBound.branch row12809_small_covers row12809_block_005_covers

def row12809_large_left_right_right : List Nat := row12809_block_007 ++ row12809_block_008
theorem row12809_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12809 row12809_large_left_right_right :=
  Budget.RowUpperBound.branch row12809_block_007_covers row12809_block_008_covers

def row12809_large_left_right : List Nat := row12809_block_006 ++ row12809_large_left_right_right
theorem row12809_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12809 row12809_large_left_right :=
  Budget.RowUpperBound.branch row12809_block_006_covers row12809_large_left_right_right_covers

def row12809_large_left : List Nat := row12809_large_left_left ++ row12809_large_left_right
theorem row12809_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12809 row12809_large_left :=
  Budget.RowUpperBound.branch row12809_large_left_left_covers row12809_large_left_right_covers

def row12809_large_right_left : List Nat := row12809_block_009 ++ row12809_block_010
theorem row12809_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12809 row12809_large_right_left :=
  Budget.RowUpperBound.branch row12809_block_009_covers row12809_block_010_covers

def row12809_large_right_right_right : List Nat := row12809_block_012 ++ row12809_block_013
theorem row12809_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12809 row12809_large_right_right_right :=
  Budget.RowUpperBound.branch row12809_block_012_covers row12809_block_013_covers

def row12809_large_right_right : List Nat := row12809_block_011 ++ row12809_large_right_right_right
theorem row12809_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12809 row12809_large_right_right :=
  Budget.RowUpperBound.branch row12809_block_011_covers row12809_large_right_right_right_covers

def row12809_large_right : List Nat := row12809_large_right_left ++ row12809_large_right_right
theorem row12809_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12809 row12809_large_right :=
  Budget.RowUpperBound.branch row12809_large_right_left_covers row12809_large_right_right_covers

def row12809_large : List Nat := row12809_large_left ++ row12809_large_right
theorem row12809_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12809 row12809_large :=
  Budget.RowUpperBound.branch row12809_large_left_covers row12809_large_right_covers

def row12809 : List Nat := row12809_large
theorem row12809_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12809 row12809 := row12809_large_covers

def row12821_small : List Nat := []
theorem row12821_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12821 row12821_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12821_block_005 : List Nat := []
theorem row12821_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12821 row12821_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12821_block_006 : List Nat := []
theorem row12821_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12821 row12821_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12821_block_007 : List Nat := [13163]
def plan12821_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12821_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12821 row12821_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12821_block_007
  · decide
  · rfl

def row12821_block_008 : List Nat := []
theorem row12821_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12821 row12821_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12821_block_009 : List Nat := []
theorem row12821_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12821 row12821_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12821_block_010 : List Nat := []
theorem row12821_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12821 row12821_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12821_block_011 : List Nat := []
theorem row12821_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12821 row12821_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12821_block_012 : List Nat := []
theorem row12821_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12821 row12821_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12821_block_013 : List Nat := []
theorem row12821_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12821 row12821_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12821_large_left_left : List Nat := row12821_small ++ row12821_block_005
theorem row12821_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12821 row12821_large_left_left :=
  Budget.RowUpperBound.branch row12821_small_covers row12821_block_005_covers

def row12821_large_left_right_right : List Nat := row12821_block_007 ++ row12821_block_008
theorem row12821_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12821 row12821_large_left_right_right :=
  Budget.RowUpperBound.branch row12821_block_007_covers row12821_block_008_covers

def row12821_large_left_right : List Nat := row12821_block_006 ++ row12821_large_left_right_right
theorem row12821_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12821 row12821_large_left_right :=
  Budget.RowUpperBound.branch row12821_block_006_covers row12821_large_left_right_right_covers

def row12821_large_left : List Nat := row12821_large_left_left ++ row12821_large_left_right
theorem row12821_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12821 row12821_large_left :=
  Budget.RowUpperBound.branch row12821_large_left_left_covers row12821_large_left_right_covers

def row12821_large_right_left : List Nat := row12821_block_009 ++ row12821_block_010
theorem row12821_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12821 row12821_large_right_left :=
  Budget.RowUpperBound.branch row12821_block_009_covers row12821_block_010_covers

def row12821_large_right_right_right : List Nat := row12821_block_012 ++ row12821_block_013
theorem row12821_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12821 row12821_large_right_right_right :=
  Budget.RowUpperBound.branch row12821_block_012_covers row12821_block_013_covers

def row12821_large_right_right : List Nat := row12821_block_011 ++ row12821_large_right_right_right
theorem row12821_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12821 row12821_large_right_right :=
  Budget.RowUpperBound.branch row12821_block_011_covers row12821_large_right_right_right_covers

def row12821_large_right : List Nat := row12821_large_right_left ++ row12821_large_right_right
theorem row12821_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12821 row12821_large_right :=
  Budget.RowUpperBound.branch row12821_large_right_left_covers row12821_large_right_right_covers

def row12821_large : List Nat := row12821_large_left ++ row12821_large_right
theorem row12821_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12821 row12821_large :=
  Budget.RowUpperBound.branch row12821_large_left_covers row12821_large_right_covers

def row12821 : List Nat := row12821_large
theorem row12821_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12821 row12821 := row12821_large_covers

def row12823_small : List Nat := []
theorem row12823_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12823 row12823_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12823_block_005 : List Nat := []
theorem row12823_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12823 row12823_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12823_block_006 : List Nat := []
theorem row12823_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12823 row12823_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12823_block_007 : List Nat := []
def plan12823_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), (.secondFactor 7), .second, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 17), (.firstFactor 37), .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 37), .first, (.secondFactor 23), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12823_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12823 row12823_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12823_block_007
  · decide
  · rfl

def row12823_block_008 : List Nat := []
theorem row12823_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12823 row12823_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12823_block_009 : List Nat := []
theorem row12823_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12823 row12823_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12823_block_010 : List Nat := []
theorem row12823_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12823 row12823_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12823_block_011 : List Nat := []
theorem row12823_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12823 row12823_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12823_block_012 : List Nat := []
theorem row12823_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12823 row12823_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12823_block_013 : List Nat := []
theorem row12823_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12823 row12823_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12823_large_left_left : List Nat := row12823_small ++ row12823_block_005
theorem row12823_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12823 row12823_large_left_left :=
  Budget.RowUpperBound.branch row12823_small_covers row12823_block_005_covers

def row12823_large_left_right_right : List Nat := row12823_block_007 ++ row12823_block_008
theorem row12823_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12823 row12823_large_left_right_right :=
  Budget.RowUpperBound.branch row12823_block_007_covers row12823_block_008_covers

def row12823_large_left_right : List Nat := row12823_block_006 ++ row12823_large_left_right_right
theorem row12823_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12823 row12823_large_left_right :=
  Budget.RowUpperBound.branch row12823_block_006_covers row12823_large_left_right_right_covers

def row12823_large_left : List Nat := row12823_large_left_left ++ row12823_large_left_right
theorem row12823_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12823 row12823_large_left :=
  Budget.RowUpperBound.branch row12823_large_left_left_covers row12823_large_left_right_covers

def row12823_large_right_left : List Nat := row12823_block_009 ++ row12823_block_010
theorem row12823_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12823 row12823_large_right_left :=
  Budget.RowUpperBound.branch row12823_block_009_covers row12823_block_010_covers

def row12823_large_right_right_right : List Nat := row12823_block_012 ++ row12823_block_013
theorem row12823_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12823 row12823_large_right_right_right :=
  Budget.RowUpperBound.branch row12823_block_012_covers row12823_block_013_covers

def row12823_large_right_right : List Nat := row12823_block_011 ++ row12823_large_right_right_right
theorem row12823_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12823 row12823_large_right_right :=
  Budget.RowUpperBound.branch row12823_block_011_covers row12823_large_right_right_right_covers

def row12823_large_right : List Nat := row12823_large_right_left ++ row12823_large_right_right
theorem row12823_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12823 row12823_large_right :=
  Budget.RowUpperBound.branch row12823_large_right_left_covers row12823_large_right_right_covers

def row12823_large : List Nat := row12823_large_left ++ row12823_large_right
theorem row12823_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12823 row12823_large :=
  Budget.RowUpperBound.branch row12823_large_left_covers row12823_large_right_covers

def row12823 : List Nat := row12823_large
theorem row12823_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12823 row12823 := row12823_large_covers

def row12829_small : List Nat := []
theorem row12829_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12829 row12829_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12829_block_005 : List Nat := []
theorem row12829_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12829 row12829_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12829_block_006 : List Nat := []
theorem row12829_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12829 row12829_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12829_block_007 : List Nat := []
def plan12829_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), (.secondFactor 17), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12829_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12829 row12829_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12829_block_007
  · decide
  · rfl

def row12829_block_008 : List Nat := []
theorem row12829_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12829 row12829_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12829_block_009 : List Nat := []
theorem row12829_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12829 row12829_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12829_block_010 : List Nat := []
theorem row12829_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12829 row12829_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12829_block_011 : List Nat := []
theorem row12829_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12829 row12829_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12829_block_012 : List Nat := []
theorem row12829_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12829 row12829_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12829_block_013 : List Nat := []
theorem row12829_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12829 row12829_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12829_large_left_left : List Nat := row12829_small ++ row12829_block_005
theorem row12829_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12829 row12829_large_left_left :=
  Budget.RowUpperBound.branch row12829_small_covers row12829_block_005_covers

def row12829_large_left_right_right : List Nat := row12829_block_007 ++ row12829_block_008
theorem row12829_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12829 row12829_large_left_right_right :=
  Budget.RowUpperBound.branch row12829_block_007_covers row12829_block_008_covers

def row12829_large_left_right : List Nat := row12829_block_006 ++ row12829_large_left_right_right
theorem row12829_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12829 row12829_large_left_right :=
  Budget.RowUpperBound.branch row12829_block_006_covers row12829_large_left_right_right_covers

def row12829_large_left : List Nat := row12829_large_left_left ++ row12829_large_left_right
theorem row12829_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12829 row12829_large_left :=
  Budget.RowUpperBound.branch row12829_large_left_left_covers row12829_large_left_right_covers

def row12829_large_right_left : List Nat := row12829_block_009 ++ row12829_block_010
theorem row12829_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12829 row12829_large_right_left :=
  Budget.RowUpperBound.branch row12829_block_009_covers row12829_block_010_covers

def row12829_large_right_right_right : List Nat := row12829_block_012 ++ row12829_block_013
theorem row12829_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12829 row12829_large_right_right_right :=
  Budget.RowUpperBound.branch row12829_block_012_covers row12829_block_013_covers

def row12829_large_right_right : List Nat := row12829_block_011 ++ row12829_large_right_right_right
theorem row12829_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12829 row12829_large_right_right :=
  Budget.RowUpperBound.branch row12829_block_011_covers row12829_large_right_right_right_covers

def row12829_large_right : List Nat := row12829_large_right_left ++ row12829_large_right_right
theorem row12829_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12829 row12829_large_right :=
  Budget.RowUpperBound.branch row12829_large_right_left_covers row12829_large_right_right_covers

def row12829_large : List Nat := row12829_large_left ++ row12829_large_right
theorem row12829_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12829 row12829_large :=
  Budget.RowUpperBound.branch row12829_large_left_covers row12829_large_right_covers

def row12829 : List Nat := row12829_large
theorem row12829_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12829 row12829 := row12829_large_covers

end ProjectEulerSolutions.PrimePairLarge
