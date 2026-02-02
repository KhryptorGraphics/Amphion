"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Music, Mic2 } from "lucide-react";
import Link from "next/link";

const svcModels = [
  {
    id: "vevosing",
    name: "VevoSing",
    description: "Singing voice conversion using flow matching - convert singer voice while preserving melody",
    icon: Music,
    href: "/svc/vevosing",
  },
];

export default function SVCPage() {
  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">Singing Voice Conversion</h1>
        <p className="text-muted-foreground">
          Convert singing voices while preserving melody and pitch
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {svcModels.map((model) => {
          const Icon = model.icon;
          return (
            <Link key={model.id} href={model.href}>
              <Card className="hover:border-primary transition-colors cursor-pointer h-full">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Icon className="h-5 w-5" />
                    {model.name}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    {model.description}
                  </p>
                  <div className="flex items-center gap-1 mt-4 text-sm text-primary">
                    Try it <ArrowRight className="h-4 w-4" />
                  </div>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>

      <Card className="border-dashed mt-6">
        <CardContent className="py-8 text-center">
          <Mic2 className="mx-auto h-12 w-12 text-muted-foreground/50 mb-4" />
          <p className="text-muted-foreground">
            More SVC models coming soon: DiffComoSVC, TransformerSVC, VitsSVC, MultipleContentsSVC
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
