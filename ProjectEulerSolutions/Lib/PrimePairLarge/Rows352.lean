import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11779_small : List Nat := []
theorem row11779_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11779 row11779_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11779_block_005 : List Nat := []
theorem row11779_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11779 row11779_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11779_block_006 : List Nat := [11953]
def plan11779_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 23), .first]
set_option maxRecDepth 2048 in
theorem row11779_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11779 row11779_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11779_block_006
  · decide
  · rfl

def row11779_block_007 : List Nat := [12577, 13183, 13591, 13687, 13903]
def plan11779_block_007 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 19), .first, .possible, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 23), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 19), .first, .second, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11779_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11779 row11779_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11779_block_007
  · decide
  · rfl

def row11779_block_008 : List Nat := []
theorem row11779_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11779 row11779_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11779_block_009 : List Nat := []
theorem row11779_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11779 row11779_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11779_block_010 : List Nat := []
theorem row11779_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11779 row11779_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11779_block_011 : List Nat := []
theorem row11779_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11779 row11779_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11779_block_012 : List Nat := []
theorem row11779_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11779 row11779_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11779_block_013 : List Nat := []
theorem row11779_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11779 row11779_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11779_large_left_left : List Nat := row11779_small ++ row11779_block_005
theorem row11779_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11779 row11779_large_left_left :=
  Budget.RowUpperBound.branch row11779_small_covers row11779_block_005_covers

def row11779_large_left_right_right : List Nat := row11779_block_007 ++ row11779_block_008
theorem row11779_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11779 row11779_large_left_right_right :=
  Budget.RowUpperBound.branch row11779_block_007_covers row11779_block_008_covers

def row11779_large_left_right : List Nat := row11779_block_006 ++ row11779_large_left_right_right
theorem row11779_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11779 row11779_large_left_right :=
  Budget.RowUpperBound.branch row11779_block_006_covers row11779_large_left_right_right_covers

def row11779_large_left : List Nat := row11779_large_left_left ++ row11779_large_left_right
theorem row11779_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11779 row11779_large_left :=
  Budget.RowUpperBound.branch row11779_large_left_left_covers row11779_large_left_right_covers

def row11779_large_right_left : List Nat := row11779_block_009 ++ row11779_block_010
theorem row11779_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11779 row11779_large_right_left :=
  Budget.RowUpperBound.branch row11779_block_009_covers row11779_block_010_covers

def row11779_large_right_right_right : List Nat := row11779_block_012 ++ row11779_block_013
theorem row11779_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11779 row11779_large_right_right_right :=
  Budget.RowUpperBound.branch row11779_block_012_covers row11779_block_013_covers

def row11779_large_right_right : List Nat := row11779_block_011 ++ row11779_large_right_right_right
theorem row11779_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11779 row11779_large_right_right :=
  Budget.RowUpperBound.branch row11779_block_011_covers row11779_large_right_right_right_covers

def row11779_large_right : List Nat := row11779_large_right_left ++ row11779_large_right_right
theorem row11779_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11779 row11779_large_right :=
  Budget.RowUpperBound.branch row11779_large_right_left_covers row11779_large_right_right_covers

def row11779_large : List Nat := row11779_large_left ++ row11779_large_right
theorem row11779_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11779 row11779_large :=
  Budget.RowUpperBound.branch row11779_large_left_covers row11779_large_right_covers

def row11779 : List Nat := row11779_large
theorem row11779_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11779 row11779 := row11779_large_covers

def row11783_small : List Nat := []
theorem row11783_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11783 row11783_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11783_block_005 : List Nat := []
theorem row11783_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11783 row11783_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11783_block_006 : List Nat := [11867]
def plan11783_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 29), .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 13), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 31), .second, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row11783_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11783 row11783_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11783_block_006
  · decide
  · rfl

def row11783_block_007 : List Nat := [13421, 13463]
def plan11783_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 13), .second, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 37), .first, .first, .first, .second, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 23), .first, (.firstFactor 19), .second, .first, (.secondFactor 7), (.firstFactor 13), .second, (.firstFactor 17), .first, .first, .first, .first, .second, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .second, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 13), .possible, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 17), .second, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11783_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11783 row11783_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11783_block_007
  · decide
  · rfl

def row11783_block_008 : List Nat := []
theorem row11783_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11783 row11783_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11783_block_009 : List Nat := []
theorem row11783_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11783 row11783_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11783_block_010 : List Nat := []
theorem row11783_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11783 row11783_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11783_block_011 : List Nat := []
theorem row11783_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11783 row11783_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11783_block_012 : List Nat := []
theorem row11783_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11783 row11783_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11783_block_013 : List Nat := []
theorem row11783_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11783 row11783_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11783_large_left_left : List Nat := row11783_small ++ row11783_block_005
theorem row11783_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11783 row11783_large_left_left :=
  Budget.RowUpperBound.branch row11783_small_covers row11783_block_005_covers

def row11783_large_left_right_right : List Nat := row11783_block_007 ++ row11783_block_008
theorem row11783_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11783 row11783_large_left_right_right :=
  Budget.RowUpperBound.branch row11783_block_007_covers row11783_block_008_covers

def row11783_large_left_right : List Nat := row11783_block_006 ++ row11783_large_left_right_right
theorem row11783_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11783 row11783_large_left_right :=
  Budget.RowUpperBound.branch row11783_block_006_covers row11783_large_left_right_right_covers

def row11783_large_left : List Nat := row11783_large_left_left ++ row11783_large_left_right
theorem row11783_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11783 row11783_large_left :=
  Budget.RowUpperBound.branch row11783_large_left_left_covers row11783_large_left_right_covers

def row11783_large_right_left : List Nat := row11783_block_009 ++ row11783_block_010
theorem row11783_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11783 row11783_large_right_left :=
  Budget.RowUpperBound.branch row11783_block_009_covers row11783_block_010_covers

def row11783_large_right_right_right : List Nat := row11783_block_012 ++ row11783_block_013
theorem row11783_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11783 row11783_large_right_right_right :=
  Budget.RowUpperBound.branch row11783_block_012_covers row11783_block_013_covers

def row11783_large_right_right : List Nat := row11783_block_011 ++ row11783_large_right_right_right
theorem row11783_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11783 row11783_large_right_right :=
  Budget.RowUpperBound.branch row11783_block_011_covers row11783_large_right_right_right_covers

def row11783_large_right : List Nat := row11783_large_right_left ++ row11783_large_right_right
theorem row11783_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11783 row11783_large_right :=
  Budget.RowUpperBound.branch row11783_large_right_left_covers row11783_large_right_right_covers

def row11783_large : List Nat := row11783_large_left ++ row11783_large_right
theorem row11783_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11783 row11783_large :=
  Budget.RowUpperBound.branch row11783_large_left_covers row11783_large_right_covers

def row11783 : List Nat := row11783_large
theorem row11783_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11783 row11783 := row11783_large_covers

def row11789_small : List Nat := []
theorem row11789_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11789 row11789_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11789_block_005 : List Nat := []
theorem row11789_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11789 row11789_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11789_block_006 : List Nat := [11933, 12269]
def plan11789_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .possible, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 31), (.firstFactor 7), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 29), .possible, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 19), .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11789_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11789 row11789_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11789_block_006
  · decide
  · rfl

def row11789_block_007 : List Nat := [13469]
def plan11789_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), (.secondFactor 23), (.firstFactor 31), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 31), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 7), .possible, .first, (.secondFactor 19), (.firstFactor 37), .first, (.secondFactor 29), .first, (.firstFactor 23), .first, (.firstFactor 19), .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 31), (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11789_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11789 row11789_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11789_block_007
  · decide
  · rfl

def row11789_block_008 : List Nat := []
theorem row11789_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11789 row11789_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11789_block_009 : List Nat := []
theorem row11789_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11789 row11789_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11789_block_010 : List Nat := []
theorem row11789_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11789 row11789_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11789_block_011 : List Nat := []
theorem row11789_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11789 row11789_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11789_block_012 : List Nat := []
theorem row11789_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11789 row11789_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11789_block_013 : List Nat := []
theorem row11789_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11789 row11789_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11789_large_left_left : List Nat := row11789_small ++ row11789_block_005
theorem row11789_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11789 row11789_large_left_left :=
  Budget.RowUpperBound.branch row11789_small_covers row11789_block_005_covers

def row11789_large_left_right_right : List Nat := row11789_block_007 ++ row11789_block_008
theorem row11789_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11789 row11789_large_left_right_right :=
  Budget.RowUpperBound.branch row11789_block_007_covers row11789_block_008_covers

def row11789_large_left_right : List Nat := row11789_block_006 ++ row11789_large_left_right_right
theorem row11789_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11789 row11789_large_left_right :=
  Budget.RowUpperBound.branch row11789_block_006_covers row11789_large_left_right_right_covers

def row11789_large_left : List Nat := row11789_large_left_left ++ row11789_large_left_right
theorem row11789_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11789 row11789_large_left :=
  Budget.RowUpperBound.branch row11789_large_left_left_covers row11789_large_left_right_covers

def row11789_large_right_left : List Nat := row11789_block_009 ++ row11789_block_010
theorem row11789_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11789 row11789_large_right_left :=
  Budget.RowUpperBound.branch row11789_block_009_covers row11789_block_010_covers

def row11789_large_right_right_right : List Nat := row11789_block_012 ++ row11789_block_013
theorem row11789_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11789 row11789_large_right_right_right :=
  Budget.RowUpperBound.branch row11789_block_012_covers row11789_block_013_covers

def row11789_large_right_right : List Nat := row11789_block_011 ++ row11789_large_right_right_right
theorem row11789_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11789 row11789_large_right_right :=
  Budget.RowUpperBound.branch row11789_block_011_covers row11789_large_right_right_right_covers

def row11789_large_right : List Nat := row11789_large_right_left ++ row11789_large_right_right
theorem row11789_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11789 row11789_large_right :=
  Budget.RowUpperBound.branch row11789_large_right_left_covers row11789_large_right_right_covers

def row11789_large : List Nat := row11789_large_left ++ row11789_large_right
theorem row11789_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11789 row11789_large :=
  Budget.RowUpperBound.branch row11789_large_left_covers row11789_large_right_covers

def row11789 : List Nat := row11789_large
theorem row11789_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11789 row11789 := row11789_large_covers

def row11801_small : List Nat := []
theorem row11801_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11801 row11801_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11801_block_005 : List Nat := []
theorem row11801_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11801 row11801_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11801_block_006 : List Nat := []
def plan11801_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 37), (.secondFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 29), .first, .second, .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row11801_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11801 row11801_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11801_block_006
  · decide
  · rfl

def row11801_block_007 : List Nat := [13001, 13577, 13763]
def plan11801_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), (.secondFactor 17), .first, .first, (.firstFactor 11), .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 23), .first, .first, (.firstFactor 29), (.secondFactor 13), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 37), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 17), .second, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 7), .first, .second, .first, (.secondFactor 17), .first, .second, (.firstFactor 31), .first, (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .second, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11801_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11801 row11801_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11801_block_007
  · decide
  · rfl

def row11801_block_008 : List Nat := []
theorem row11801_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11801 row11801_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11801_block_009 : List Nat := []
theorem row11801_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11801 row11801_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11801_block_010 : List Nat := []
theorem row11801_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11801 row11801_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11801_block_011 : List Nat := []
theorem row11801_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11801 row11801_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11801_block_012 : List Nat := []
theorem row11801_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11801 row11801_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11801_block_013 : List Nat := []
theorem row11801_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11801 row11801_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11801_large_left_left : List Nat := row11801_small ++ row11801_block_005
theorem row11801_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11801 row11801_large_left_left :=
  Budget.RowUpperBound.branch row11801_small_covers row11801_block_005_covers

def row11801_large_left_right_right : List Nat := row11801_block_007 ++ row11801_block_008
theorem row11801_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11801 row11801_large_left_right_right :=
  Budget.RowUpperBound.branch row11801_block_007_covers row11801_block_008_covers

def row11801_large_left_right : List Nat := row11801_block_006 ++ row11801_large_left_right_right
theorem row11801_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11801 row11801_large_left_right :=
  Budget.RowUpperBound.branch row11801_block_006_covers row11801_large_left_right_right_covers

def row11801_large_left : List Nat := row11801_large_left_left ++ row11801_large_left_right
theorem row11801_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11801 row11801_large_left :=
  Budget.RowUpperBound.branch row11801_large_left_left_covers row11801_large_left_right_covers

def row11801_large_right_left : List Nat := row11801_block_009 ++ row11801_block_010
theorem row11801_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11801 row11801_large_right_left :=
  Budget.RowUpperBound.branch row11801_block_009_covers row11801_block_010_covers

def row11801_large_right_right_right : List Nat := row11801_block_012 ++ row11801_block_013
theorem row11801_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11801 row11801_large_right_right_right :=
  Budget.RowUpperBound.branch row11801_block_012_covers row11801_block_013_covers

def row11801_large_right_right : List Nat := row11801_block_011 ++ row11801_large_right_right_right
theorem row11801_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11801 row11801_large_right_right :=
  Budget.RowUpperBound.branch row11801_block_011_covers row11801_large_right_right_right_covers

def row11801_large_right : List Nat := row11801_large_right_left ++ row11801_large_right_right
theorem row11801_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11801 row11801_large_right :=
  Budget.RowUpperBound.branch row11801_large_right_left_covers row11801_large_right_right_covers

def row11801_large : List Nat := row11801_large_left ++ row11801_large_right
theorem row11801_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11801 row11801_large :=
  Budget.RowUpperBound.branch row11801_large_left_covers row11801_large_right_covers

def row11801 : List Nat := row11801_large
theorem row11801_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11801 row11801 := row11801_large_covers

end ProjectEulerSolutions.PrimePairLarge
