import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12583_small : List Nat := []
theorem row12583_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12583 row12583_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12583_block_005 : List Nat := []
theorem row12583_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12583 row12583_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12583_block_006 : List Nat := []
theorem row12583_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12583 row12583_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12583_block_007 : List Nat := [13291]
def plan12583_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 23), (.firstFactor 19), (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .second, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 19), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12583_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12583 row12583_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12583_block_007
  · decide
  · rfl

def row12583_block_008 : List Nat := []
theorem row12583_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12583 row12583_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12583_block_009 : List Nat := []
theorem row12583_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12583 row12583_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12583_block_010 : List Nat := []
theorem row12583_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12583 row12583_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12583_block_011 : List Nat := []
theorem row12583_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12583 row12583_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12583_block_012 : List Nat := []
theorem row12583_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12583 row12583_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12583_block_013 : List Nat := []
theorem row12583_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12583 row12583_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12583_large_left_left : List Nat := row12583_small ++ row12583_block_005
theorem row12583_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12583 row12583_large_left_left :=
  Budget.RowUpperBound.branch row12583_small_covers row12583_block_005_covers

def row12583_large_left_right_right : List Nat := row12583_block_007 ++ row12583_block_008
theorem row12583_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12583 row12583_large_left_right_right :=
  Budget.RowUpperBound.branch row12583_block_007_covers row12583_block_008_covers

def row12583_large_left_right : List Nat := row12583_block_006 ++ row12583_large_left_right_right
theorem row12583_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12583 row12583_large_left_right :=
  Budget.RowUpperBound.branch row12583_block_006_covers row12583_large_left_right_right_covers

def row12583_large_left : List Nat := row12583_large_left_left ++ row12583_large_left_right
theorem row12583_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12583 row12583_large_left :=
  Budget.RowUpperBound.branch row12583_large_left_left_covers row12583_large_left_right_covers

def row12583_large_right_left : List Nat := row12583_block_009 ++ row12583_block_010
theorem row12583_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12583 row12583_large_right_left :=
  Budget.RowUpperBound.branch row12583_block_009_covers row12583_block_010_covers

def row12583_large_right_right_right : List Nat := row12583_block_012 ++ row12583_block_013
theorem row12583_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12583 row12583_large_right_right_right :=
  Budget.RowUpperBound.branch row12583_block_012_covers row12583_block_013_covers

def row12583_large_right_right : List Nat := row12583_block_011 ++ row12583_large_right_right_right
theorem row12583_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12583 row12583_large_right_right :=
  Budget.RowUpperBound.branch row12583_block_011_covers row12583_large_right_right_right_covers

def row12583_large_right : List Nat := row12583_large_right_left ++ row12583_large_right_right
theorem row12583_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12583 row12583_large_right :=
  Budget.RowUpperBound.branch row12583_large_right_left_covers row12583_large_right_right_covers

def row12583_large : List Nat := row12583_large_left ++ row12583_large_right
theorem row12583_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12583 row12583_large :=
  Budget.RowUpperBound.branch row12583_large_left_covers row12583_large_right_covers

def row12583 : List Nat := row12583_large
theorem row12583_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12583 row12583 := row12583_large_covers

def row12589_small : List Nat := []
theorem row12589_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12589 row12589_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12589_block_005 : List Nat := []
theorem row12589_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12589 row12589_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12589_block_006 : List Nat := []
theorem row12589_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12589 row12589_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12589_block_007 : List Nat := []
def plan12589_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 37), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 17), .second, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12589_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12589 row12589_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12589_block_007
  · decide
  · rfl

def row12589_block_008 : List Nat := []
theorem row12589_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12589 row12589_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12589_block_009 : List Nat := []
theorem row12589_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12589 row12589_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12589_block_010 : List Nat := []
theorem row12589_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12589 row12589_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12589_block_011 : List Nat := []
theorem row12589_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12589 row12589_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12589_block_012 : List Nat := []
theorem row12589_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12589 row12589_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12589_block_013 : List Nat := []
theorem row12589_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12589 row12589_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12589_large_left_left : List Nat := row12589_small ++ row12589_block_005
theorem row12589_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12589 row12589_large_left_left :=
  Budget.RowUpperBound.branch row12589_small_covers row12589_block_005_covers

def row12589_large_left_right_right : List Nat := row12589_block_007 ++ row12589_block_008
theorem row12589_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12589 row12589_large_left_right_right :=
  Budget.RowUpperBound.branch row12589_block_007_covers row12589_block_008_covers

def row12589_large_left_right : List Nat := row12589_block_006 ++ row12589_large_left_right_right
theorem row12589_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12589 row12589_large_left_right :=
  Budget.RowUpperBound.branch row12589_block_006_covers row12589_large_left_right_right_covers

def row12589_large_left : List Nat := row12589_large_left_left ++ row12589_large_left_right
theorem row12589_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12589 row12589_large_left :=
  Budget.RowUpperBound.branch row12589_large_left_left_covers row12589_large_left_right_covers

def row12589_large_right_left : List Nat := row12589_block_009 ++ row12589_block_010
theorem row12589_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12589 row12589_large_right_left :=
  Budget.RowUpperBound.branch row12589_block_009_covers row12589_block_010_covers

def row12589_large_right_right_right : List Nat := row12589_block_012 ++ row12589_block_013
theorem row12589_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12589 row12589_large_right_right_right :=
  Budget.RowUpperBound.branch row12589_block_012_covers row12589_block_013_covers

def row12589_large_right_right : List Nat := row12589_block_011 ++ row12589_large_right_right_right
theorem row12589_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12589 row12589_large_right_right :=
  Budget.RowUpperBound.branch row12589_block_011_covers row12589_large_right_right_right_covers

def row12589_large_right : List Nat := row12589_large_right_left ++ row12589_large_right_right
theorem row12589_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12589 row12589_large_right :=
  Budget.RowUpperBound.branch row12589_large_right_left_covers row12589_large_right_right_covers

def row12589_large : List Nat := row12589_large_left ++ row12589_large_right
theorem row12589_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12589 row12589_large :=
  Budget.RowUpperBound.branch row12589_large_left_covers row12589_large_right_covers

def row12589 : List Nat := row12589_large
theorem row12589_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12589 row12589 := row12589_large_covers

def row12601_small : List Nat := []
theorem row12601_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12601 row12601_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12601_block_005 : List Nat := []
theorem row12601_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12601 row12601_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12601_block_006 : List Nat := []
theorem row12601_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12601 row12601_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12601_block_007 : List Nat := [12853]
def plan12601_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 37), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 37), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12601_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12601 row12601_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12601_block_007
  · decide
  · rfl

def row12601_block_008 : List Nat := []
theorem row12601_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12601 row12601_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12601_block_009 : List Nat := []
theorem row12601_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12601 row12601_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12601_block_010 : List Nat := []
theorem row12601_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12601 row12601_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12601_block_011 : List Nat := []
theorem row12601_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12601 row12601_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12601_block_012 : List Nat := []
theorem row12601_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12601 row12601_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12601_block_013 : List Nat := []
theorem row12601_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12601 row12601_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12601_large_left_left : List Nat := row12601_small ++ row12601_block_005
theorem row12601_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12601 row12601_large_left_left :=
  Budget.RowUpperBound.branch row12601_small_covers row12601_block_005_covers

def row12601_large_left_right_right : List Nat := row12601_block_007 ++ row12601_block_008
theorem row12601_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12601 row12601_large_left_right_right :=
  Budget.RowUpperBound.branch row12601_block_007_covers row12601_block_008_covers

def row12601_large_left_right : List Nat := row12601_block_006 ++ row12601_large_left_right_right
theorem row12601_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12601 row12601_large_left_right :=
  Budget.RowUpperBound.branch row12601_block_006_covers row12601_large_left_right_right_covers

def row12601_large_left : List Nat := row12601_large_left_left ++ row12601_large_left_right
theorem row12601_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12601 row12601_large_left :=
  Budget.RowUpperBound.branch row12601_large_left_left_covers row12601_large_left_right_covers

def row12601_large_right_left : List Nat := row12601_block_009 ++ row12601_block_010
theorem row12601_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12601 row12601_large_right_left :=
  Budget.RowUpperBound.branch row12601_block_009_covers row12601_block_010_covers

def row12601_large_right_right_right : List Nat := row12601_block_012 ++ row12601_block_013
theorem row12601_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12601 row12601_large_right_right_right :=
  Budget.RowUpperBound.branch row12601_block_012_covers row12601_block_013_covers

def row12601_large_right_right : List Nat := row12601_block_011 ++ row12601_large_right_right_right
theorem row12601_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12601 row12601_large_right_right :=
  Budget.RowUpperBound.branch row12601_block_011_covers row12601_large_right_right_right_covers

def row12601_large_right : List Nat := row12601_large_right_left ++ row12601_large_right_right
theorem row12601_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12601 row12601_large_right :=
  Budget.RowUpperBound.branch row12601_large_right_left_covers row12601_large_right_right_covers

def row12601_large : List Nat := row12601_large_left ++ row12601_large_right
theorem row12601_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12601 row12601_large :=
  Budget.RowUpperBound.branch row12601_large_left_covers row12601_large_right_covers

def row12601 : List Nat := row12601_large
theorem row12601_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12601 row12601 := row12601_large_covers

def row12611_small : List Nat := []
theorem row12611_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12611 row12611_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12611_block_005 : List Nat := []
theorem row12611_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12611 row12611_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12611_block_006 : List Nat := []
theorem row12611_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12611 row12611_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12611_block_007 : List Nat := [13241]
def plan12611_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, .second, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 37), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .second, (.firstFactor 7), .first, .second, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12611_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12611 row12611_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12611_block_007
  · decide
  · rfl

def row12611_block_008 : List Nat := []
theorem row12611_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12611 row12611_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12611_block_009 : List Nat := []
theorem row12611_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12611 row12611_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12611_block_010 : List Nat := []
theorem row12611_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12611 row12611_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12611_block_011 : List Nat := []
theorem row12611_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12611 row12611_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12611_block_012 : List Nat := []
theorem row12611_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12611 row12611_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12611_block_013 : List Nat := []
theorem row12611_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12611 row12611_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12611_large_left_left : List Nat := row12611_small ++ row12611_block_005
theorem row12611_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12611 row12611_large_left_left :=
  Budget.RowUpperBound.branch row12611_small_covers row12611_block_005_covers

def row12611_large_left_right_right : List Nat := row12611_block_007 ++ row12611_block_008
theorem row12611_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12611 row12611_large_left_right_right :=
  Budget.RowUpperBound.branch row12611_block_007_covers row12611_block_008_covers

def row12611_large_left_right : List Nat := row12611_block_006 ++ row12611_large_left_right_right
theorem row12611_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12611 row12611_large_left_right :=
  Budget.RowUpperBound.branch row12611_block_006_covers row12611_large_left_right_right_covers

def row12611_large_left : List Nat := row12611_large_left_left ++ row12611_large_left_right
theorem row12611_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12611 row12611_large_left :=
  Budget.RowUpperBound.branch row12611_large_left_left_covers row12611_large_left_right_covers

def row12611_large_right_left : List Nat := row12611_block_009 ++ row12611_block_010
theorem row12611_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12611 row12611_large_right_left :=
  Budget.RowUpperBound.branch row12611_block_009_covers row12611_block_010_covers

def row12611_large_right_right_right : List Nat := row12611_block_012 ++ row12611_block_013
theorem row12611_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12611 row12611_large_right_right_right :=
  Budget.RowUpperBound.branch row12611_block_012_covers row12611_block_013_covers

def row12611_large_right_right : List Nat := row12611_block_011 ++ row12611_large_right_right_right
theorem row12611_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12611 row12611_large_right_right :=
  Budget.RowUpperBound.branch row12611_block_011_covers row12611_large_right_right_right_covers

def row12611_large_right : List Nat := row12611_large_right_left ++ row12611_large_right_right
theorem row12611_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12611 row12611_large_right :=
  Budget.RowUpperBound.branch row12611_large_right_left_covers row12611_large_right_right_covers

def row12611_large : List Nat := row12611_large_left ++ row12611_large_right
theorem row12611_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12611 row12611_large :=
  Budget.RowUpperBound.branch row12611_large_left_covers row12611_large_right_covers

def row12611 : List Nat := row12611_large
theorem row12611_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12611 row12611 := row12611_large_covers

end ProjectEulerSolutions.PrimePairLarge
