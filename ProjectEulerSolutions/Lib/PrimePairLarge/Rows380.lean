import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12763_small : List Nat := []
theorem row12763_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12763 row12763_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12763_block_005 : List Nat := []
theorem row12763_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12763 row12763_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12763_block_006 : List Nat := []
theorem row12763_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12763 row12763_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12763_block_007 : List Nat := []
def plan12763_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12763_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12763 row12763_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12763_block_007
  · decide
  · rfl

def row12763_block_008 : List Nat := []
theorem row12763_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12763 row12763_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12763_block_009 : List Nat := []
theorem row12763_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12763 row12763_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12763_block_010 : List Nat := []
theorem row12763_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12763 row12763_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12763_block_011 : List Nat := []
theorem row12763_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12763 row12763_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12763_block_012 : List Nat := []
theorem row12763_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12763 row12763_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12763_block_013 : List Nat := []
theorem row12763_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12763 row12763_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12763_large_left_left : List Nat := row12763_small ++ row12763_block_005
theorem row12763_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12763 row12763_large_left_left :=
  Budget.RowUpperBound.branch row12763_small_covers row12763_block_005_covers

def row12763_large_left_right_right : List Nat := row12763_block_007 ++ row12763_block_008
theorem row12763_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12763 row12763_large_left_right_right :=
  Budget.RowUpperBound.branch row12763_block_007_covers row12763_block_008_covers

def row12763_large_left_right : List Nat := row12763_block_006 ++ row12763_large_left_right_right
theorem row12763_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12763 row12763_large_left_right :=
  Budget.RowUpperBound.branch row12763_block_006_covers row12763_large_left_right_right_covers

def row12763_large_left : List Nat := row12763_large_left_left ++ row12763_large_left_right
theorem row12763_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12763 row12763_large_left :=
  Budget.RowUpperBound.branch row12763_large_left_left_covers row12763_large_left_right_covers

def row12763_large_right_left : List Nat := row12763_block_009 ++ row12763_block_010
theorem row12763_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12763 row12763_large_right_left :=
  Budget.RowUpperBound.branch row12763_block_009_covers row12763_block_010_covers

def row12763_large_right_right_right : List Nat := row12763_block_012 ++ row12763_block_013
theorem row12763_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12763 row12763_large_right_right_right :=
  Budget.RowUpperBound.branch row12763_block_012_covers row12763_block_013_covers

def row12763_large_right_right : List Nat := row12763_block_011 ++ row12763_large_right_right_right
theorem row12763_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12763 row12763_large_right_right :=
  Budget.RowUpperBound.branch row12763_block_011_covers row12763_large_right_right_right_covers

def row12763_large_right : List Nat := row12763_large_right_left ++ row12763_large_right_right
theorem row12763_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12763 row12763_large_right :=
  Budget.RowUpperBound.branch row12763_large_right_left_covers row12763_large_right_right_covers

def row12763_large : List Nat := row12763_large_left ++ row12763_large_right
theorem row12763_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12763 row12763_large :=
  Budget.RowUpperBound.branch row12763_large_left_covers row12763_large_right_covers

def row12763 : List Nat := row12763_large
theorem row12763_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12763 row12763 := row12763_large_covers

def row12781_small : List Nat := []
theorem row12781_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12781 row12781_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12781_block_005 : List Nat := []
theorem row12781_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12781 row12781_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12781_block_006 : List Nat := []
theorem row12781_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12781 row12781_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12781_block_007 : List Nat := [13003]
def plan12781_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), (.secondFactor 13), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), .second, (.firstFactor 11), .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12781_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12781 row12781_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12781_block_007
  · decide
  · rfl

def row12781_block_008 : List Nat := []
theorem row12781_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12781 row12781_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12781_block_009 : List Nat := []
theorem row12781_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12781 row12781_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12781_block_010 : List Nat := []
theorem row12781_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12781 row12781_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12781_block_011 : List Nat := []
theorem row12781_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12781 row12781_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12781_block_012 : List Nat := []
theorem row12781_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12781 row12781_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12781_block_013 : List Nat := []
theorem row12781_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12781 row12781_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12781_large_left_left : List Nat := row12781_small ++ row12781_block_005
theorem row12781_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12781 row12781_large_left_left :=
  Budget.RowUpperBound.branch row12781_small_covers row12781_block_005_covers

def row12781_large_left_right_right : List Nat := row12781_block_007 ++ row12781_block_008
theorem row12781_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12781 row12781_large_left_right_right :=
  Budget.RowUpperBound.branch row12781_block_007_covers row12781_block_008_covers

def row12781_large_left_right : List Nat := row12781_block_006 ++ row12781_large_left_right_right
theorem row12781_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12781 row12781_large_left_right :=
  Budget.RowUpperBound.branch row12781_block_006_covers row12781_large_left_right_right_covers

def row12781_large_left : List Nat := row12781_large_left_left ++ row12781_large_left_right
theorem row12781_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12781 row12781_large_left :=
  Budget.RowUpperBound.branch row12781_large_left_left_covers row12781_large_left_right_covers

def row12781_large_right_left : List Nat := row12781_block_009 ++ row12781_block_010
theorem row12781_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12781 row12781_large_right_left :=
  Budget.RowUpperBound.branch row12781_block_009_covers row12781_block_010_covers

def row12781_large_right_right_right : List Nat := row12781_block_012 ++ row12781_block_013
theorem row12781_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12781 row12781_large_right_right_right :=
  Budget.RowUpperBound.branch row12781_block_012_covers row12781_block_013_covers

def row12781_large_right_right : List Nat := row12781_block_011 ++ row12781_large_right_right_right
theorem row12781_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12781 row12781_large_right_right :=
  Budget.RowUpperBound.branch row12781_block_011_covers row12781_large_right_right_right_covers

def row12781_large_right : List Nat := row12781_large_right_left ++ row12781_large_right_right
theorem row12781_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12781 row12781_large_right :=
  Budget.RowUpperBound.branch row12781_large_right_left_covers row12781_large_right_right_covers

def row12781_large : List Nat := row12781_large_left ++ row12781_large_right
theorem row12781_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12781 row12781_large :=
  Budget.RowUpperBound.branch row12781_large_left_covers row12781_large_right_covers

def row12781 : List Nat := row12781_large
theorem row12781_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12781 row12781 := row12781_large_covers

def row12791_small : List Nat := []
theorem row12791_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12791 row12791_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12791_block_005 : List Nat := []
theorem row12791_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12791 row12791_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12791_block_006 : List Nat := []
theorem row12791_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12791 row12791_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12791_block_007 : List Nat := [12911]
def plan12791_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, .possible, (.firstFactor 31), .first, (.firstFactor 11), (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .second, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12791_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12791 row12791_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12791_block_007
  · decide
  · rfl

def row12791_block_008 : List Nat := []
theorem row12791_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12791 row12791_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12791_block_009 : List Nat := []
theorem row12791_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12791 row12791_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12791_block_010 : List Nat := []
theorem row12791_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12791 row12791_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12791_block_011 : List Nat := []
theorem row12791_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12791 row12791_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12791_block_012 : List Nat := []
theorem row12791_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12791 row12791_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12791_block_013 : List Nat := []
theorem row12791_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12791 row12791_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12791_large_left_left : List Nat := row12791_small ++ row12791_block_005
theorem row12791_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12791 row12791_large_left_left :=
  Budget.RowUpperBound.branch row12791_small_covers row12791_block_005_covers

def row12791_large_left_right_right : List Nat := row12791_block_007 ++ row12791_block_008
theorem row12791_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12791 row12791_large_left_right_right :=
  Budget.RowUpperBound.branch row12791_block_007_covers row12791_block_008_covers

def row12791_large_left_right : List Nat := row12791_block_006 ++ row12791_large_left_right_right
theorem row12791_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12791 row12791_large_left_right :=
  Budget.RowUpperBound.branch row12791_block_006_covers row12791_large_left_right_right_covers

def row12791_large_left : List Nat := row12791_large_left_left ++ row12791_large_left_right
theorem row12791_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12791 row12791_large_left :=
  Budget.RowUpperBound.branch row12791_large_left_left_covers row12791_large_left_right_covers

def row12791_large_right_left : List Nat := row12791_block_009 ++ row12791_block_010
theorem row12791_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12791 row12791_large_right_left :=
  Budget.RowUpperBound.branch row12791_block_009_covers row12791_block_010_covers

def row12791_large_right_right_right : List Nat := row12791_block_012 ++ row12791_block_013
theorem row12791_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12791 row12791_large_right_right_right :=
  Budget.RowUpperBound.branch row12791_block_012_covers row12791_block_013_covers

def row12791_large_right_right : List Nat := row12791_block_011 ++ row12791_large_right_right_right
theorem row12791_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12791 row12791_large_right_right :=
  Budget.RowUpperBound.branch row12791_block_011_covers row12791_large_right_right_right_covers

def row12791_large_right : List Nat := row12791_large_right_left ++ row12791_large_right_right
theorem row12791_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12791 row12791_large_right :=
  Budget.RowUpperBound.branch row12791_large_right_left_covers row12791_large_right_right_covers

def row12791_large : List Nat := row12791_large_left ++ row12791_large_right
theorem row12791_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12791 row12791_large :=
  Budget.RowUpperBound.branch row12791_large_left_covers row12791_large_right_covers

def row12791 : List Nat := row12791_large
theorem row12791_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12791 row12791 := row12791_large_covers

def row12799_small : List Nat := []
theorem row12799_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12799 row12799_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12799_block_005 : List Nat := []
theorem row12799_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12799 row12799_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12799_block_006 : List Nat := []
theorem row12799_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12799 row12799_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12799_block_007 : List Nat := []
def plan12799_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 31), (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, (.secondFactor 19), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .second, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12799_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12799 row12799_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12799_block_007
  · decide
  · rfl

def row12799_block_008 : List Nat := []
theorem row12799_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12799 row12799_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12799_block_009 : List Nat := []
theorem row12799_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12799 row12799_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12799_block_010 : List Nat := []
theorem row12799_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12799 row12799_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12799_block_011 : List Nat := []
theorem row12799_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12799 row12799_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12799_block_012 : List Nat := []
theorem row12799_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12799 row12799_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12799_block_013 : List Nat := []
theorem row12799_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12799 row12799_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12799_large_left_left : List Nat := row12799_small ++ row12799_block_005
theorem row12799_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12799 row12799_large_left_left :=
  Budget.RowUpperBound.branch row12799_small_covers row12799_block_005_covers

def row12799_large_left_right_right : List Nat := row12799_block_007 ++ row12799_block_008
theorem row12799_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12799 row12799_large_left_right_right :=
  Budget.RowUpperBound.branch row12799_block_007_covers row12799_block_008_covers

def row12799_large_left_right : List Nat := row12799_block_006 ++ row12799_large_left_right_right
theorem row12799_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12799 row12799_large_left_right :=
  Budget.RowUpperBound.branch row12799_block_006_covers row12799_large_left_right_right_covers

def row12799_large_left : List Nat := row12799_large_left_left ++ row12799_large_left_right
theorem row12799_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12799 row12799_large_left :=
  Budget.RowUpperBound.branch row12799_large_left_left_covers row12799_large_left_right_covers

def row12799_large_right_left : List Nat := row12799_block_009 ++ row12799_block_010
theorem row12799_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12799 row12799_large_right_left :=
  Budget.RowUpperBound.branch row12799_block_009_covers row12799_block_010_covers

def row12799_large_right_right_right : List Nat := row12799_block_012 ++ row12799_block_013
theorem row12799_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12799 row12799_large_right_right_right :=
  Budget.RowUpperBound.branch row12799_block_012_covers row12799_block_013_covers

def row12799_large_right_right : List Nat := row12799_block_011 ++ row12799_large_right_right_right
theorem row12799_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12799 row12799_large_right_right :=
  Budget.RowUpperBound.branch row12799_block_011_covers row12799_large_right_right_right_covers

def row12799_large_right : List Nat := row12799_large_right_left ++ row12799_large_right_right
theorem row12799_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12799 row12799_large_right :=
  Budget.RowUpperBound.branch row12799_large_right_left_covers row12799_large_right_right_covers

def row12799_large : List Nat := row12799_large_left ++ row12799_large_right
theorem row12799_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12799 row12799_large :=
  Budget.RowUpperBound.branch row12799_large_left_covers row12799_large_right_covers

def row12799 : List Nat := row12799_large
theorem row12799_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12799 row12799 := row12799_large_covers

end ProjectEulerSolutions.PrimePairLarge
