import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12721_small : List Nat := []
theorem row12721_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12721 row12721_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12721_block_005 : List Nat := []
theorem row12721_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12721 row12721_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12721_block_006 : List Nat := []
theorem row12721_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12721 row12721_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12721_block_007 : List Nat := [12889]
def plan12721_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .second, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 23), (.firstFactor 13), (.secondFactor 29), (.firstFactor 11), .possible, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12721_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12721 row12721_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12721_block_007
  · decide
  · rfl

def row12721_block_008 : List Nat := []
theorem row12721_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12721 row12721_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12721_block_009 : List Nat := []
theorem row12721_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12721 row12721_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12721_block_010 : List Nat := []
theorem row12721_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12721 row12721_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12721_block_011 : List Nat := []
theorem row12721_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12721 row12721_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12721_block_012 : List Nat := []
theorem row12721_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12721 row12721_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12721_block_013 : List Nat := []
theorem row12721_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12721 row12721_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12721_large_left_left : List Nat := row12721_small ++ row12721_block_005
theorem row12721_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12721 row12721_large_left_left :=
  Budget.RowUpperBound.branch row12721_small_covers row12721_block_005_covers

def row12721_large_left_right_right : List Nat := row12721_block_007 ++ row12721_block_008
theorem row12721_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12721 row12721_large_left_right_right :=
  Budget.RowUpperBound.branch row12721_block_007_covers row12721_block_008_covers

def row12721_large_left_right : List Nat := row12721_block_006 ++ row12721_large_left_right_right
theorem row12721_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12721 row12721_large_left_right :=
  Budget.RowUpperBound.branch row12721_block_006_covers row12721_large_left_right_right_covers

def row12721_large_left : List Nat := row12721_large_left_left ++ row12721_large_left_right
theorem row12721_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12721 row12721_large_left :=
  Budget.RowUpperBound.branch row12721_large_left_left_covers row12721_large_left_right_covers

def row12721_large_right_left : List Nat := row12721_block_009 ++ row12721_block_010
theorem row12721_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12721 row12721_large_right_left :=
  Budget.RowUpperBound.branch row12721_block_009_covers row12721_block_010_covers

def row12721_large_right_right_right : List Nat := row12721_block_012 ++ row12721_block_013
theorem row12721_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12721 row12721_large_right_right_right :=
  Budget.RowUpperBound.branch row12721_block_012_covers row12721_block_013_covers

def row12721_large_right_right : List Nat := row12721_block_011 ++ row12721_large_right_right_right
theorem row12721_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12721 row12721_large_right_right :=
  Budget.RowUpperBound.branch row12721_block_011_covers row12721_large_right_right_right_covers

def row12721_large_right : List Nat := row12721_large_right_left ++ row12721_large_right_right
theorem row12721_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12721 row12721_large_right :=
  Budget.RowUpperBound.branch row12721_large_right_left_covers row12721_large_right_right_covers

def row12721_large : List Nat := row12721_large_left ++ row12721_large_right
theorem row12721_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12721 row12721_large :=
  Budget.RowUpperBound.branch row12721_large_left_covers row12721_large_right_covers

def row12721 : List Nat := row12721_large
theorem row12721_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12721 row12721 := row12721_large_covers

def row12739_small : List Nat := []
theorem row12739_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12739 row12739_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12739_block_005 : List Nat := []
theorem row12739_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12739 row12739_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12739_block_006 : List Nat := []
theorem row12739_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12739 row12739_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12739_block_007 : List Nat := [12889, 13063]
def plan12739_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 31), .possible, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .possible, (.secondFactor 7), (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12739_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12739 row12739_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12739_block_007
  · decide
  · rfl

def row12739_block_008 : List Nat := []
theorem row12739_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12739 row12739_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12739_block_009 : List Nat := []
theorem row12739_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12739 row12739_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12739_block_010 : List Nat := []
theorem row12739_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12739 row12739_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12739_block_011 : List Nat := []
theorem row12739_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12739 row12739_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12739_block_012 : List Nat := []
theorem row12739_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12739 row12739_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12739_block_013 : List Nat := []
theorem row12739_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12739 row12739_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12739_large_left_left : List Nat := row12739_small ++ row12739_block_005
theorem row12739_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12739 row12739_large_left_left :=
  Budget.RowUpperBound.branch row12739_small_covers row12739_block_005_covers

def row12739_large_left_right_right : List Nat := row12739_block_007 ++ row12739_block_008
theorem row12739_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12739 row12739_large_left_right_right :=
  Budget.RowUpperBound.branch row12739_block_007_covers row12739_block_008_covers

def row12739_large_left_right : List Nat := row12739_block_006 ++ row12739_large_left_right_right
theorem row12739_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12739 row12739_large_left_right :=
  Budget.RowUpperBound.branch row12739_block_006_covers row12739_large_left_right_right_covers

def row12739_large_left : List Nat := row12739_large_left_left ++ row12739_large_left_right
theorem row12739_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12739 row12739_large_left :=
  Budget.RowUpperBound.branch row12739_large_left_left_covers row12739_large_left_right_covers

def row12739_large_right_left : List Nat := row12739_block_009 ++ row12739_block_010
theorem row12739_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12739 row12739_large_right_left :=
  Budget.RowUpperBound.branch row12739_block_009_covers row12739_block_010_covers

def row12739_large_right_right_right : List Nat := row12739_block_012 ++ row12739_block_013
theorem row12739_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12739 row12739_large_right_right_right :=
  Budget.RowUpperBound.branch row12739_block_012_covers row12739_block_013_covers

def row12739_large_right_right : List Nat := row12739_block_011 ++ row12739_large_right_right_right
theorem row12739_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12739 row12739_large_right_right :=
  Budget.RowUpperBound.branch row12739_block_011_covers row12739_large_right_right_right_covers

def row12739_large_right : List Nat := row12739_large_right_left ++ row12739_large_right_right
theorem row12739_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12739 row12739_large_right :=
  Budget.RowUpperBound.branch row12739_large_right_left_covers row12739_large_right_right_covers

def row12739_large : List Nat := row12739_large_left ++ row12739_large_right
theorem row12739_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12739 row12739_large :=
  Budget.RowUpperBound.branch row12739_large_left_covers row12739_large_right_covers

def row12739 : List Nat := row12739_large
theorem row12739_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12739 row12739 := row12739_large_covers

def row12743_small : List Nat := []
theorem row12743_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12743 row12743_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12743_block_005 : List Nat := []
theorem row12743_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12743 row12743_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12743_block_006 : List Nat := []
theorem row12743_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12743 row12743_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12743_block_007 : List Nat := [13049]
def plan12743_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 31), .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12743_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12743 row12743_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12743_block_007
  · decide
  · rfl

def row12743_block_008 : List Nat := []
theorem row12743_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12743 row12743_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12743_block_009 : List Nat := []
theorem row12743_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12743 row12743_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12743_block_010 : List Nat := []
theorem row12743_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12743 row12743_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12743_block_011 : List Nat := []
theorem row12743_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12743 row12743_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12743_block_012 : List Nat := []
theorem row12743_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12743 row12743_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12743_block_013 : List Nat := []
theorem row12743_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12743 row12743_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12743_large_left_left : List Nat := row12743_small ++ row12743_block_005
theorem row12743_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12743 row12743_large_left_left :=
  Budget.RowUpperBound.branch row12743_small_covers row12743_block_005_covers

def row12743_large_left_right_right : List Nat := row12743_block_007 ++ row12743_block_008
theorem row12743_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12743 row12743_large_left_right_right :=
  Budget.RowUpperBound.branch row12743_block_007_covers row12743_block_008_covers

def row12743_large_left_right : List Nat := row12743_block_006 ++ row12743_large_left_right_right
theorem row12743_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12743 row12743_large_left_right :=
  Budget.RowUpperBound.branch row12743_block_006_covers row12743_large_left_right_right_covers

def row12743_large_left : List Nat := row12743_large_left_left ++ row12743_large_left_right
theorem row12743_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12743 row12743_large_left :=
  Budget.RowUpperBound.branch row12743_large_left_left_covers row12743_large_left_right_covers

def row12743_large_right_left : List Nat := row12743_block_009 ++ row12743_block_010
theorem row12743_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12743 row12743_large_right_left :=
  Budget.RowUpperBound.branch row12743_block_009_covers row12743_block_010_covers

def row12743_large_right_right_right : List Nat := row12743_block_012 ++ row12743_block_013
theorem row12743_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12743 row12743_large_right_right_right :=
  Budget.RowUpperBound.branch row12743_block_012_covers row12743_block_013_covers

def row12743_large_right_right : List Nat := row12743_block_011 ++ row12743_large_right_right_right
theorem row12743_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12743 row12743_large_right_right :=
  Budget.RowUpperBound.branch row12743_block_011_covers row12743_large_right_right_right_covers

def row12743_large_right : List Nat := row12743_large_right_left ++ row12743_large_right_right
theorem row12743_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12743 row12743_large_right :=
  Budget.RowUpperBound.branch row12743_large_right_left_covers row12743_large_right_right_covers

def row12743_large : List Nat := row12743_large_left ++ row12743_large_right
theorem row12743_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12743 row12743_large :=
  Budget.RowUpperBound.branch row12743_large_left_covers row12743_large_right_covers

def row12743 : List Nat := row12743_large
theorem row12743_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12743 row12743 := row12743_large_covers

def row12757_small : List Nat := []
theorem row12757_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12757 row12757_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12757_block_005 : List Nat := []
theorem row12757_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12757 row12757_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12757_block_006 : List Nat := []
theorem row12757_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12757 row12757_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12757_block_007 : List Nat := []
def plan12757_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12757_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12757 row12757_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12757_block_007
  · decide
  · rfl

def row12757_block_008 : List Nat := []
theorem row12757_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12757 row12757_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12757_block_009 : List Nat := []
theorem row12757_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12757 row12757_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12757_block_010 : List Nat := []
theorem row12757_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12757 row12757_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12757_block_011 : List Nat := []
theorem row12757_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12757 row12757_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12757_block_012 : List Nat := []
theorem row12757_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12757 row12757_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12757_block_013 : List Nat := []
theorem row12757_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12757 row12757_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12757_large_left_left : List Nat := row12757_small ++ row12757_block_005
theorem row12757_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12757 row12757_large_left_left :=
  Budget.RowUpperBound.branch row12757_small_covers row12757_block_005_covers

def row12757_large_left_right_right : List Nat := row12757_block_007 ++ row12757_block_008
theorem row12757_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12757 row12757_large_left_right_right :=
  Budget.RowUpperBound.branch row12757_block_007_covers row12757_block_008_covers

def row12757_large_left_right : List Nat := row12757_block_006 ++ row12757_large_left_right_right
theorem row12757_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12757 row12757_large_left_right :=
  Budget.RowUpperBound.branch row12757_block_006_covers row12757_large_left_right_right_covers

def row12757_large_left : List Nat := row12757_large_left_left ++ row12757_large_left_right
theorem row12757_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12757 row12757_large_left :=
  Budget.RowUpperBound.branch row12757_large_left_left_covers row12757_large_left_right_covers

def row12757_large_right_left : List Nat := row12757_block_009 ++ row12757_block_010
theorem row12757_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12757 row12757_large_right_left :=
  Budget.RowUpperBound.branch row12757_block_009_covers row12757_block_010_covers

def row12757_large_right_right_right : List Nat := row12757_block_012 ++ row12757_block_013
theorem row12757_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12757 row12757_large_right_right_right :=
  Budget.RowUpperBound.branch row12757_block_012_covers row12757_block_013_covers

def row12757_large_right_right : List Nat := row12757_block_011 ++ row12757_large_right_right_right
theorem row12757_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12757 row12757_large_right_right :=
  Budget.RowUpperBound.branch row12757_block_011_covers row12757_large_right_right_right_covers

def row12757_large_right : List Nat := row12757_large_right_left ++ row12757_large_right_right
theorem row12757_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12757 row12757_large_right :=
  Budget.RowUpperBound.branch row12757_large_right_left_covers row12757_large_right_right_covers

def row12757_large : List Nat := row12757_large_left ++ row12757_large_right
theorem row12757_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12757 row12757_large :=
  Budget.RowUpperBound.branch row12757_large_left_covers row12757_large_right_covers

def row12757 : List Nat := row12757_large
theorem row12757_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12757 row12757 := row12757_large_covers

end ProjectEulerSolutions.PrimePairLarge
