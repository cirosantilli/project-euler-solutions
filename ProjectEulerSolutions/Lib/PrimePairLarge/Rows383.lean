import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12899_small : List Nat := []
theorem row12899_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12899 row12899_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12899_block_005 : List Nat := []
theorem row12899_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12899 row12899_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12899_block_006 : List Nat := []
theorem row12899_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12899 row12899_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12899_block_007 : List Nat := [12959]
def plan12899_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 7), (.firstFactor 29), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12899_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12899 row12899_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12899_block_007
  · decide
  · rfl

def row12899_block_008 : List Nat := []
theorem row12899_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12899 row12899_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12899_block_009 : List Nat := []
theorem row12899_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12899 row12899_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12899_block_010 : List Nat := []
theorem row12899_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12899 row12899_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12899_block_011 : List Nat := []
theorem row12899_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12899 row12899_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12899_block_012 : List Nat := []
theorem row12899_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12899 row12899_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12899_block_013 : List Nat := []
theorem row12899_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12899 row12899_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12899_large_left_left : List Nat := row12899_small ++ row12899_block_005
theorem row12899_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12899 row12899_large_left_left :=
  Budget.RowUpperBound.branch row12899_small_covers row12899_block_005_covers

def row12899_large_left_right_right : List Nat := row12899_block_007 ++ row12899_block_008
theorem row12899_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12899 row12899_large_left_right_right :=
  Budget.RowUpperBound.branch row12899_block_007_covers row12899_block_008_covers

def row12899_large_left_right : List Nat := row12899_block_006 ++ row12899_large_left_right_right
theorem row12899_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12899 row12899_large_left_right :=
  Budget.RowUpperBound.branch row12899_block_006_covers row12899_large_left_right_right_covers

def row12899_large_left : List Nat := row12899_large_left_left ++ row12899_large_left_right
theorem row12899_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12899 row12899_large_left :=
  Budget.RowUpperBound.branch row12899_large_left_left_covers row12899_large_left_right_covers

def row12899_large_right_left : List Nat := row12899_block_009 ++ row12899_block_010
theorem row12899_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12899 row12899_large_right_left :=
  Budget.RowUpperBound.branch row12899_block_009_covers row12899_block_010_covers

def row12899_large_right_right_right : List Nat := row12899_block_012 ++ row12899_block_013
theorem row12899_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12899 row12899_large_right_right_right :=
  Budget.RowUpperBound.branch row12899_block_012_covers row12899_block_013_covers

def row12899_large_right_right : List Nat := row12899_block_011 ++ row12899_large_right_right_right
theorem row12899_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12899 row12899_large_right_right :=
  Budget.RowUpperBound.branch row12899_block_011_covers row12899_large_right_right_right_covers

def row12899_large_right : List Nat := row12899_large_right_left ++ row12899_large_right_right
theorem row12899_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12899 row12899_large_right :=
  Budget.RowUpperBound.branch row12899_large_right_left_covers row12899_large_right_right_covers

def row12899_large : List Nat := row12899_large_left ++ row12899_large_right
theorem row12899_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12899 row12899_large :=
  Budget.RowUpperBound.branch row12899_large_left_covers row12899_large_right_covers

def row12899 : List Nat := row12899_large
theorem row12899_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12899 row12899 := row12899_large_covers

def row12907_small : List Nat := []
theorem row12907_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12907 row12907_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12907_block_005 : List Nat := []
theorem row12907_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12907 row12907_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12907_block_006 : List Nat := []
theorem row12907_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12907 row12907_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12907_block_007 : List Nat := []
def plan12907_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12907_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12907 row12907_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12907_block_007
  · decide
  · rfl

def row12907_block_008 : List Nat := []
theorem row12907_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12907 row12907_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12907_block_009 : List Nat := []
theorem row12907_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12907 row12907_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12907_block_010 : List Nat := []
theorem row12907_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12907 row12907_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12907_block_011 : List Nat := []
theorem row12907_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12907 row12907_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12907_block_012 : List Nat := []
theorem row12907_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12907 row12907_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12907_block_013 : List Nat := []
theorem row12907_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12907 row12907_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12907_large_left_left : List Nat := row12907_small ++ row12907_block_005
theorem row12907_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12907 row12907_large_left_left :=
  Budget.RowUpperBound.branch row12907_small_covers row12907_block_005_covers

def row12907_large_left_right_right : List Nat := row12907_block_007 ++ row12907_block_008
theorem row12907_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12907 row12907_large_left_right_right :=
  Budget.RowUpperBound.branch row12907_block_007_covers row12907_block_008_covers

def row12907_large_left_right : List Nat := row12907_block_006 ++ row12907_large_left_right_right
theorem row12907_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12907 row12907_large_left_right :=
  Budget.RowUpperBound.branch row12907_block_006_covers row12907_large_left_right_right_covers

def row12907_large_left : List Nat := row12907_large_left_left ++ row12907_large_left_right
theorem row12907_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12907 row12907_large_left :=
  Budget.RowUpperBound.branch row12907_large_left_left_covers row12907_large_left_right_covers

def row12907_large_right_left : List Nat := row12907_block_009 ++ row12907_block_010
theorem row12907_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12907 row12907_large_right_left :=
  Budget.RowUpperBound.branch row12907_block_009_covers row12907_block_010_covers

def row12907_large_right_right_right : List Nat := row12907_block_012 ++ row12907_block_013
theorem row12907_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12907 row12907_large_right_right_right :=
  Budget.RowUpperBound.branch row12907_block_012_covers row12907_block_013_covers

def row12907_large_right_right : List Nat := row12907_block_011 ++ row12907_large_right_right_right
theorem row12907_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12907 row12907_large_right_right :=
  Budget.RowUpperBound.branch row12907_block_011_covers row12907_large_right_right_right_covers

def row12907_large_right : List Nat := row12907_large_right_left ++ row12907_large_right_right
theorem row12907_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12907 row12907_large_right :=
  Budget.RowUpperBound.branch row12907_large_right_left_covers row12907_large_right_right_covers

def row12907_large : List Nat := row12907_large_left ++ row12907_large_right
theorem row12907_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12907 row12907_large :=
  Budget.RowUpperBound.branch row12907_large_left_covers row12907_large_right_covers

def row12907 : List Nat := row12907_large
theorem row12907_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12907 row12907 := row12907_large_covers

def row12911_small : List Nat := []
theorem row12911_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12911 row12911_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12911_block_005 : List Nat := []
theorem row12911_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12911 row12911_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12911_block_006 : List Nat := []
theorem row12911_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12911 row12911_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12911_block_007 : List Nat := []
def plan12911_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 13), .second, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 19), (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12911_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12911 row12911_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12911_block_007
  · decide
  · rfl

def row12911_block_008 : List Nat := []
theorem row12911_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12911 row12911_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12911_block_009 : List Nat := []
theorem row12911_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12911 row12911_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12911_block_010 : List Nat := []
theorem row12911_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12911 row12911_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12911_block_011 : List Nat := []
theorem row12911_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12911 row12911_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12911_block_012 : List Nat := []
theorem row12911_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12911 row12911_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12911_block_013 : List Nat := []
theorem row12911_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12911 row12911_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12911_large_left_left : List Nat := row12911_small ++ row12911_block_005
theorem row12911_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12911 row12911_large_left_left :=
  Budget.RowUpperBound.branch row12911_small_covers row12911_block_005_covers

def row12911_large_left_right_right : List Nat := row12911_block_007 ++ row12911_block_008
theorem row12911_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12911 row12911_large_left_right_right :=
  Budget.RowUpperBound.branch row12911_block_007_covers row12911_block_008_covers

def row12911_large_left_right : List Nat := row12911_block_006 ++ row12911_large_left_right_right
theorem row12911_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12911 row12911_large_left_right :=
  Budget.RowUpperBound.branch row12911_block_006_covers row12911_large_left_right_right_covers

def row12911_large_left : List Nat := row12911_large_left_left ++ row12911_large_left_right
theorem row12911_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12911 row12911_large_left :=
  Budget.RowUpperBound.branch row12911_large_left_left_covers row12911_large_left_right_covers

def row12911_large_right_left : List Nat := row12911_block_009 ++ row12911_block_010
theorem row12911_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12911 row12911_large_right_left :=
  Budget.RowUpperBound.branch row12911_block_009_covers row12911_block_010_covers

def row12911_large_right_right_right : List Nat := row12911_block_012 ++ row12911_block_013
theorem row12911_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12911 row12911_large_right_right_right :=
  Budget.RowUpperBound.branch row12911_block_012_covers row12911_block_013_covers

def row12911_large_right_right : List Nat := row12911_block_011 ++ row12911_large_right_right_right
theorem row12911_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12911 row12911_large_right_right :=
  Budget.RowUpperBound.branch row12911_block_011_covers row12911_large_right_right_right_covers

def row12911_large_right : List Nat := row12911_large_right_left ++ row12911_large_right_right
theorem row12911_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12911 row12911_large_right :=
  Budget.RowUpperBound.branch row12911_large_right_left_covers row12911_large_right_right_covers

def row12911_large : List Nat := row12911_large_left ++ row12911_large_right
theorem row12911_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12911 row12911_large :=
  Budget.RowUpperBound.branch row12911_large_left_covers row12911_large_right_covers

def row12911 : List Nat := row12911_large
theorem row12911_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12911 row12911 := row12911_large_covers

def row12917_small : List Nat := []
theorem row12917_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12917 row12917_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12917_block_005 : List Nat := []
theorem row12917_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12917 row12917_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12917_block_006 : List Nat := []
theorem row12917_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12917 row12917_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12917_block_007 : List Nat := [12953]
def plan12917_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), (.secondFactor 23), .possible, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12917_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12917 row12917_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12917_block_007
  · decide
  · rfl

def row12917_block_008 : List Nat := []
theorem row12917_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12917 row12917_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12917_block_009 : List Nat := []
theorem row12917_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12917 row12917_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12917_block_010 : List Nat := []
theorem row12917_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12917 row12917_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12917_block_011 : List Nat := []
theorem row12917_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12917 row12917_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12917_block_012 : List Nat := []
theorem row12917_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12917 row12917_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12917_block_013 : List Nat := []
theorem row12917_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12917 row12917_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12917_large_left_left : List Nat := row12917_small ++ row12917_block_005
theorem row12917_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12917 row12917_large_left_left :=
  Budget.RowUpperBound.branch row12917_small_covers row12917_block_005_covers

def row12917_large_left_right_right : List Nat := row12917_block_007 ++ row12917_block_008
theorem row12917_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12917 row12917_large_left_right_right :=
  Budget.RowUpperBound.branch row12917_block_007_covers row12917_block_008_covers

def row12917_large_left_right : List Nat := row12917_block_006 ++ row12917_large_left_right_right
theorem row12917_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12917 row12917_large_left_right :=
  Budget.RowUpperBound.branch row12917_block_006_covers row12917_large_left_right_right_covers

def row12917_large_left : List Nat := row12917_large_left_left ++ row12917_large_left_right
theorem row12917_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12917 row12917_large_left :=
  Budget.RowUpperBound.branch row12917_large_left_left_covers row12917_large_left_right_covers

def row12917_large_right_left : List Nat := row12917_block_009 ++ row12917_block_010
theorem row12917_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12917 row12917_large_right_left :=
  Budget.RowUpperBound.branch row12917_block_009_covers row12917_block_010_covers

def row12917_large_right_right_right : List Nat := row12917_block_012 ++ row12917_block_013
theorem row12917_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12917 row12917_large_right_right_right :=
  Budget.RowUpperBound.branch row12917_block_012_covers row12917_block_013_covers

def row12917_large_right_right : List Nat := row12917_block_011 ++ row12917_large_right_right_right
theorem row12917_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12917 row12917_large_right_right :=
  Budget.RowUpperBound.branch row12917_block_011_covers row12917_large_right_right_right_covers

def row12917_large_right : List Nat := row12917_large_right_left ++ row12917_large_right_right
theorem row12917_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12917 row12917_large_right :=
  Budget.RowUpperBound.branch row12917_large_right_left_covers row12917_large_right_right_covers

def row12917_large : List Nat := row12917_large_left ++ row12917_large_right
theorem row12917_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12917 row12917_large :=
  Budget.RowUpperBound.branch row12917_large_left_covers row12917_large_right_covers

def row12917 : List Nat := row12917_large
theorem row12917_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12917 row12917 := row12917_large_covers

end ProjectEulerSolutions.PrimePairLarge
