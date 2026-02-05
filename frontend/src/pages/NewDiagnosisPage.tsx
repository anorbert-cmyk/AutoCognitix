import { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Wrench, HelpCircle } from 'lucide-react';

// New components
import { PageContainer, FloatingBottomBar } from '@/components/layouts';
import { WizardStepper, type WizardStep } from '@/components/composite';
import {
  DiagnosisForm,
  AnalysisProgress,
  RecentAnalysisList,
  type DiagnosisFormData,
  type AnalysisStep,
} from '@/components/features/diagnosis';
import { Button } from '@/components/lib';

// API hooks
import { useAnalyzeDiagnosis, useDiagnosisHistory } from '@/services/hooks';

type WizardStepId = 'input' | 'analysis' | 'report';

const wizardSteps: { id: WizardStepId; title: string }[] = [
  { id: 'input', title: 'Adatbevitel' },
  { id: 'analysis', title: 'Elemzés' },
  { id: 'report', title: 'Jelentés' },
];

const analysisStepsTemplate: AnalysisStep[] = [
  { id: 'vehicle', title: 'Jármű adatok ellenőrzése', status: 'pending' },
  { id: 'dtc', title: 'DTC kód azonosítása', status: 'pending' },
  { id: 'symptoms', title: 'Tünet elemzés', status: 'pending' },
  { id: 'ranking', title: 'Hibalehetőségek rangsorolása', status: 'pending' },
  { id: 'solutions', title: 'Javítási javaslatok generálása', status: 'pending' },
];

export default function NewDiagnosisPage() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState<WizardStepId>('input');
  const [analysisSteps, setAnalysisSteps] = useState<AnalysisStep[]>(
    analysisStepsTemplate.map((s) => ({ ...s }))
  );
  const [currentAnalysisStepIndex, setCurrentAnalysisStepIndex] = useState(0);
  const [diagnosisId, setDiagnosisId] = useState<string | null>(null);

  // API mutation
  const analyzeDiagnosis = useAnalyzeDiagnosis();

  // Get recent diagnoses for the floating bar
  const { data: historyData } = useDiagnosisHistory({ limit: 5 });

  const recentAnalyses = historyData?.items.map((item) => ({
    id: item.id,
    vehicleInfo: `${item.vehicle_year} ${item.vehicle_make} ${item.vehicle_model}`,
    dtcCode: item.dtc_codes[0] || 'N/A',
    timestamp: item.created_at,
  })) || [];

  // Convert wizard steps to WizardStepper format
  const getWizardSteps = (): WizardStep[] => {
    return wizardSteps.map((step, index) => {
      const currentIndex = wizardSteps.findIndex((s) => s.id === currentStep);
      return {
        title: step.title,
        status:
          index < currentIndex
            ? 'completed'
            : index === currentIndex
            ? 'active'
            : 'pending',
      };
    });
  };

  // Simulate analysis progress
  const runAnalysis = useCallback(() => {
    setCurrentStep('analysis');
    setAnalysisSteps(analysisStepsTemplate.map((s) => ({ ...s })));
    setCurrentAnalysisStepIndex(0);

    // Simulate step-by-step progress
    let stepIndex = 0;
    const interval = setInterval(() => {
      setAnalysisSteps((prevSteps) => {
        const newSteps = [...prevSteps];
        if (stepIndex > 0) {
          newSteps[stepIndex - 1].status = 'completed';
        }
        if (stepIndex < newSteps.length) {
          newSteps[stepIndex].status = 'in_progress';
          setCurrentAnalysisStepIndex(stepIndex);
        }
        return newSteps;
      });

      stepIndex++;
      if (stepIndex > analysisStepsTemplate.length) {
        clearInterval(interval);
        // Mark last step as completed
        setAnalysisSteps((prevSteps) => {
          const newSteps = [...prevSteps];
          newSteps[newSteps.length - 1].status = 'completed';
          return newSteps;
        });
      }
    }, 1500);
  }, []);

  // Handle form submission
  const handleFormSubmit = useCallback(
    async (data: DiagnosisFormData) => {
      // Start analysis animation
      runAnalysis();

      // Make actual API call
      try {
        const result = await analyzeDiagnosis.mutateAsync({
          dtcCodes: data.dtcCodes,
          vehicleMake: data.vehicleMake,
          vehicleModel: data.vehicleModel || '',
          vehicleYear: data.vehicleYear ? parseInt(data.vehicleYear) : new Date().getFullYear(),
          symptoms: data.ownerComplaints,
          additionalContext: data.mechanicNotes || undefined,
        });
        setDiagnosisId(result.id);
      } catch (error) {
        console.error('Diagnosis creation failed:', error);
        // Still navigate to mock result for demo
        setDiagnosisId('mock-id');
      }
    },
    [analyzeDiagnosis, runAnalysis]
  );

  // Handle analysis completion
  const handleAnalysisComplete = useCallback(() => {
    // Small delay before navigating
    setTimeout(() => {
      setCurrentStep('report');
      // Navigate to result page
      if (diagnosisId) {
        navigate(`/result/${diagnosisId}`);
      } else {
        // Navigate with mock data for demo
        navigate('/result/demo');
      }
    }, 500);
  }, [diagnosisId, navigate]);

  // Handle recent analysis click
  const handleRecentClick = useCallback(
    (item: { id: string }) => {
      navigate(`/result/${item.id}`);
    },
    [navigate]
  );

  // Check if analysis is complete
  useEffect(() => {
    if (currentStep === 'analysis') {
      const allCompleted = analysisSteps.every((s) => s.status === 'completed');
      if (allCompleted) {
        handleAnalysisComplete();
      }
    }
  }, [analysisSteps, currentStep, handleAnalysisComplete]);

  return (
    <>
      <PageContainer
        maxWidth="xl"
        padding="md"
        hasFloatingBar={currentStep === 'analysis'}
      >
        {/* Page Header */}
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-2">
            <Wrench className="h-8 w-8 text-primary-600" />
            <h1 className="text-2xl font-bold text-foreground">
              Új diagnosztika
            </h1>
          </div>
          <p className="text-muted-foreground">
            Adja meg a hibakódokat és jármű adatokat az AI elemzés indításához
          </p>
        </div>

        {/* Wizard Stepper */}
        <div className="mb-8">
          <WizardStepper
            steps={getWizardSteps()}
            currentStep={wizardSteps.findIndex((s) => s.id === currentStep)}
          />
        </div>

        {/* Step Content */}
        {currentStep === 'input' && (
          <DiagnosisForm
            onSubmit={handleFormSubmit}
            isSubmitting={analyzeDiagnosis.isPending}
          />
        )}

        {currentStep === 'analysis' && (
          <AnalysisProgress
            steps={analysisSteps}
            currentStep={currentAnalysisStepIndex}
            onComplete={handleAnalysisComplete}
          />
        )}
      </PageContainer>

      {/* Floating Bottom Bar - Only during analysis */}
      {currentStep === 'analysis' && (
        <FloatingBottomBar
          visible
          leftContent={
            <RecentAnalysisList
              items={recentAnalyses}
              onItemClick={handleRecentClick}
            />
          }
          rightContent={
            <Button variant="ghost" size="sm" className="text-muted-foreground">
              <HelpCircle className="h-4 w-4 mr-2" />
              Segítségre van szükséged?
            </Button>
          }
        />
      )}
    </>
  );
}
