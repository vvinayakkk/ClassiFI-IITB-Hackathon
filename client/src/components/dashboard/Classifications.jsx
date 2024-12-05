import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Drawer, DrawerClose, DrawerContent, DrawerDescription, DrawerFooter, DrawerHeader, DrawerTitle, DrawerTrigger } from "@/components/ui/drawer";
import { Info } from 'lucide-react';
import { CLASSIFICATION_DATA, CATEGORY_DETAILS } from '@/constants/dashboardData';

const Classifications = () => {
  const getCategoryDetails = (category) => CATEGORY_DETAILS[category] || {
    totalPDFs: 0,
    averageConfidence: 0,
    commonKeywords: [],
    documentTypes: "N/A",
    description: "No detailed information available.",
    typicalContent: []
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Classifications</CardTitle>
        <CardDescription>View document classification categories</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {CLASSIFICATION_DATA.map((item) => (
            <Drawer key={item.category}>
              <div className="bg-white border rounded-lg p-4 flex items-center justify-between hover:shadow-md transition-shadow">
                <div className="flex items-center space-x-4">
                  <span className="text-2xl">{item.icon}</span>
                  <div>
                    <p className="font-medium">{item.category}</p>
                    <p className="text-sm text-gray-500">{item.count} PDFs trained</p>
                  </div>
                </div>
                <DrawerTrigger asChild>
                  <button className="text-blue-600 hover:bg-blue-50 p-2 rounded-full">
                    <Info className="h-5 w-5" />
                  </button>
                </DrawerTrigger>
              </div>
              <DrawerContent>
                {(() => {
                  const details = getCategoryDetails(item.category);
                  return (
                    <>
                      <DrawerHeader>
                        <DrawerTitle>{item.category} Classification</DrawerTitle>
                        <DrawerDescription>
                          Detailed insights into {item.category} document classification
                        </DrawerDescription>
                      </DrawerHeader>
                      <div className="grid md:grid-cols-2 gap-6 p-6">
                        <div>
                          <h3 className="text-lg font-semibold mb-4">Training Metrics</h3>
                          <div className="space-y-4">
                            <div className="bg-gray-50 p-4 rounded-lg">
                              <p className="text-sm text-gray-600">Total PDFs Analyzed</p>
                              <p className="text-2xl font-bold text-blue-600">{details.totalPDFs}</p>
                            </div>
                            <div className="bg-gray-50 p-4 rounded-lg">
                              <p className="text-sm text-gray-600">Average Classification Confidence</p>
                              <p className="text-2xl font-bold text-green-600">{details.averageConfidence}%</p>
                            </div>
                          </div>
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold mb-4">Category Insights</h3>
                          <div className="space-y-4">
                            <div className="bg-gray-50 p-4 rounded-lg">
                              <p className="text-sm text-gray-600">Common Keywords</p>
                              <div className="flex flex-wrap gap-2 mt-2">
                                {details.commonKeywords?.map(keyword => (
                                  <Badge key={keyword} variant="secondary">{keyword}</Badge>
                                )) || details.topSkills?.map(skill => (
                                  <Badge key={skill} variant="secondary">{skill}</Badge>
                                ))}
                              </div>
                            </div>
                            <div className="bg-gray-50 p-4 rounded-lg">
                              <p className="text-sm text-gray-600">Document Types</p>
                              <p className="text-xl font-bold">{details.documentTypes || details.averageExperience}</p>
                            </div>
                          </div>
                        </div>
                        <div className="md:col-span-2">
                          <div className="bg-gray-50 p-4 rounded-lg">
                            <h3 className="text-lg font-semibold mb-2">Category Description</h3>
                            <p>{details.description}</p>
                          </div>
                          <div className="bg-gray-50 p-4 rounded-lg mt-4">
                            <h3 className="text-lg font-semibold mb-2">Typical Content</h3>
                            <ul className="list-disc list-inside">
                              {(details.typicalContent || details.keyResponsibilities)?.map((item, index) => (
                                <li key={index} className="text-sm">{item}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                      <DrawerFooter>
                        <DrawerClose asChild>
                          <Button variant="outline">Close</Button>
                        </DrawerClose>
                      </DrawerFooter>
                    </>
                  );
                })()}
              </DrawerContent>
            </Drawer>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default Classifications;